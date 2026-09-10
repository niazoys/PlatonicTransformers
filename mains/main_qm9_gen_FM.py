"""QM9 flow-matching training and evaluation.

This keeps the QM9 data/metrics pipeline from ``main_qm9_gen.py`` but replaces
EDM noise conditioning with straight-line conditional flow matching:
sample z0 ~ N(0, I), interpolate zt=(1-t)z0+t z1, and predict z1-z0.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Timer
from torch.optim.lr_scheduler import CosineAnnealingLR

from mains._optim import make_param_groups
from mains.main_qm9_gen import (
    collate_fn,
    load_data,
    save_test_metrics,
)
from platonic_transformers.datasets.qm9_bond_analyze import check_stability
from platonic_transformers.datasets.qm9_rdkit_utils import (
    BasicMolecularMetrics,
    ZatomMolecularMetrics,
    run_posebusters,
)
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.utils.utils import RandomSOd, subtract_mean
from torch.utils.data import DataLoader


class QM9FMModel(pl.LightningModule):
    """Node-wise conditional flow matching model for QM9 graphs."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})
        self.config = config
        use_charges = config.dataset.use_charges
        scalar_dim = 5 + (1 if use_charges else 0)
        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'")

        self.net = PlatonicTransformer(
            input_dim=scalar_dim + 1,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=scalar_dim,
            output_dim_vec=1,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            scalar_task_level="node",
            vector_task_level="node",
            ffn_readout=config.model.ffn_readout,
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
            rope_on_values=config.model.get("rope_on_values", False),
            attention_backend=config.model.get("attention_backend", "scatter"),
        )
        if config.model.get("compile", True):
            self.net = torch.compile(self.net)
        self.rotation_generator = RandomSOd(3)
        self.num_atoms_sampler = None
        self.edm_analyzer = None
        self.zatom_analyzer = None

    def set_num_atoms_sampler(self, sampler):
        self.num_atoms_sampler = sampler

    def init_molecule_analyzer(self, dataset_info, edm_smiles, zatom_smiles):
        self.edm_analyzer = BasicMolecularMetrics(dataset_info, edm_smiles)
        self.zatom_analyzer = ZatomMolecularMetrics(dataset_info, zatom_smiles)

    def _scale_x(self, x):
        x = x.clone()
        x[:, :-1] = x[:, :-1] / self.config.diffusion.normalize_x_factor
        if self.config.dataset.use_charges:
            x[:, -1] = x[:, -1] / self.config.diffusion.normalize_charge_factor
        return x

    def _velocity(self, x_t, pos_t, batch, t):
        """EDM-preconditioned flow-matching velocity field.

        Mirrors EDMPrecond's construction (platonic_transformers/diffusion/edm.py)
        input-for-input: c_in normalizes the noisy input to unit variance, c_noise
        conditions the network, and the raw output is combined via c_skip/c_out
        into the final estimate. The difference from EDMPrecond is the TARGET:
        EDM's c_skip/c_out are the variance-minimizing linear predictor of x_1 from
        x_t; here they are re-derived (same method) for target v=x_1-x_0, so the
        network still predicts a genuine flow-matching velocity rather than
        becoming an x-prediction denoiser. With sigma_data=1 this gives
        c_skip(t)=(2t-1)/(t^2+(1-t)^2), c_out(t)=c_in(t) -- see training_step.
        t is per-graph, shape (num_graphs,).
        """
        sigma_data = self.config.diffusion.sigma_data
        t_graph = t.reshape(-1, 1)
        denom = t_graph ** 2 * sigma_data ** 2 + (1 - t_graph) ** 2
        c_in = 1 / denom.sqrt()
        c_skip = (t_graph * sigma_data ** 2 - (1 - t_graph)) / denom
        c_out = sigma_data / denom.sqrt()
        c_noise = (((1 - t_graph) / t_graph).log()) / 4  # log(sigma_equiv)/4, EDM's own c_noise

        c_in_n, c_skip_n, c_out_n, c_noise_n = (v[batch] for v in (c_in, c_skip, c_out, c_noise))

        x_in = c_in_n * x_t
        pos_in = c_in_n * pos_t
        scalar_in = torch.cat([x_in, c_noise_n], dim=-1)
        scalar_out, vector_out = self.net(
            scalar_in, pos_in, batch, vec=None,
            avg_num_nodes=self.config.diffusion.avg_num_nodes,
        )
        # Unlike EDMPrecond's F = x_in - scalar_out (a denoiser-specific default:
        # zero output means "return the noisy input as data"), F here is the raw
        # output directly -- for a v-target, that makes a zero-init network's
        # prediction reduce to c_skip(t)*x_t, the exact optimal x_t-only linear
        # predictor of v at every t (verified: matches EDM's own clean gain=1 at
        # its analogous low-noise end, without the mismatch the borrowed EDM
        # convention had at t->1).
        v_x = c_skip_n * x_t + c_out_n * scalar_out
        v_pos = c_skip_n * pos_t + c_out_n * vector_out.squeeze(1)
        return v_x, v_pos

    def _sample_t(self, num_graphs, device):
        """Sample flow-matching time on the same effective schedule as EDM's
        log-normal sigma, via the SNR-matching correspondence t = 1/(1+sigma)
        used throughout (see sample()).
        """
        rnd_normal = torch.randn(num_graphs, device=device)
        sigma = (rnd_normal * self.config.diffusion.P_std + self.config.diffusion.P_mean).exp()
        return 1.0 / (1.0 + sigma)

    def _edm_weight(self, node_t):
        # weight = 1/c_out(t)^2 for the velocity target above. Bounded in
        # [0.5, 1] (never needs EDM's max_weight clamp) since c_out here never
        # shrinks to 0 the way EDM's x-target c_out does as sigma->0.
        sigma_data = self.config.diffusion.sigma_data
        return (node_t ** 2 * sigma_data ** 2 + (1 - node_t) ** 2) / sigma_data ** 2

    def training_step(self, batch, batch_idx):
        pos = subtract_mean(batch["pos"], batch["batch"])
        x = self._scale_x(batch["x"])
        if self.config.training.train_augm:
            rot = self.rotation_generator().type_as(pos)
            pos = torch.einsum("ij,bj->bi", rot, pos)
        num_graphs = int(batch["batch"].max().item()) + 1
        t = self._sample_t(num_graphs, pos.device)
        x0 = torch.randn_like(x)
        pos0 = subtract_mean(torch.randn_like(pos), batch["batch"])
        x1 = x
        pos1 = pos
        node_t = t[batch["batch"], None]
        xt = (1 - node_t) * x0 + node_t * x1
        post = (1 - node_t) * pos0 + node_t * pos1
        vx, vp = self._velocity(xt, post, batch["batch"], t)
        weight = self._edm_weight(node_t)
        loss = (weight * (vx - (x1 - x0)).square()).mean() + (weight * (vp - (pos1 - pos0)).square()).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True,
                 batch_size=num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True,
                 batch_size=int(batch["batch"].max().item()) + 1)
        return loss

    def _flow_loss(self, batch):
        pos = subtract_mean(batch["pos"], batch["batch"])
        x = self._scale_x(batch["x"])
        num_graphs = int(batch["batch"].max().item()) + 1
        t = self._sample_t(num_graphs, pos.device)
        x0, pos0 = torch.randn_like(x), subtract_mean(torch.randn_like(pos), batch["batch"])
        node_t = t[batch["batch"], None]
        vx, vp = self._velocity((1 - node_t) * x0 + node_t * x,
                                (1 - node_t) * pos0 + node_t * pos,
                                batch["batch"], t)
        weight = self._edm_weight(node_t)
        return (weight * (vx - (x - x0)).square()).mean() + (weight * (vp - (pos - pos0)).square()).mean()

    @torch.no_grad()
    def sample(self, num_molecules, num_steps=None, S_churn=None,
               S_min=0.0, S_max=float("inf"), S_noise=1.0):
        """EDM-equivalent stochastic sampler for the flow-matching path.

        Runs the same Karras sigma schedule, churn/gamma formula, and Heun
        2nd-order correction as ``edm_sampler`` (platonic_transformers/diffusion/edm.py),
        so S_churn means the same thing here as it does for the EDM model.
        EDM's VE sigma and flow-matching's t are related by the SNR-matching
        reparametrization t(sigma) = 1/(1+sigma) (see https://diffusionflow.github.io/).
        ``_velocity`` still predicts the flow-matching velocity field; the exact
        identity D(x_t,t) = x_t + (1-t)*v(x_t,t) converts it to the denoiser
        estimate the Karras-schedule arithmetic below operates on.
        """
        self.eval()
        num_steps = num_steps or self.config.diffusion.num_steps
        if S_churn is None:
            S_churn = self.config.diffusion.get("S_churn", 0.0)
        sigma_min = self.config.diffusion.sigma_min
        sigma_max = self.config.diffusion.sigma_max
        rho = self.config.diffusion.rho

        num_atoms = self.num_atoms_sampler(num_molecules).to(self.device)
        graph_ids = torch.arange(num_molecules, device=self.device)
        batch = torch.repeat_interleave(graph_ids, num_atoms)

        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

        def denoise(x_edm, pos_edm, sigma):
            t = 1.0 / (1.0 + sigma)
            x_t = t * x_edm
            pos_t = t * pos_edm
            t_full = torch.full((num_molecules,), t.item(), device=self.device)
            vx, vp = self._velocity(x_t, pos_t, batch, t_full)
            return x_t + (1 - t) * vx, pos_t + (1 - t) * vp

        x_edm = torch.randn((len(batch), 5 + int(self.config.dataset.use_charges)), device=self.device)
        pos_edm = subtract_mean(torch.randn((len(batch), 3), device=self.device), batch)

        for i in range(num_steps):
            sigma_cur, sigma_next = sigma_steps[i], sigma_steps[i + 1]
            gamma = min(S_churn / num_steps, 2**0.5 - 1) if S_min <= sigma_cur <= S_max else 0
            sigma_hat = sigma_cur * (1 + gamma)
            noise_scale = (sigma_hat**2 - sigma_cur**2).sqrt() * S_noise
            x_hat = x_edm + noise_scale * torch.randn_like(x_edm)
            pos_hat = pos_edm + noise_scale * torch.randn_like(pos_edm)

            D_x, D_pos = denoise(x_hat, pos_hat, sigma_hat)
            dx = (x_hat - D_x) / sigma_hat
            dpos = (pos_hat - D_pos) / sigma_hat
            x_edm = x_hat + (sigma_next - sigma_hat) * dx
            pos_edm = pos_hat + (sigma_next - sigma_hat) * dpos

            if i < num_steps - 1:
                D_x2, D_pos2 = denoise(x_edm, pos_edm, sigma_next)
                dx2 = (x_edm - D_x2) / sigma_next
                dpos2 = (pos_edm - D_pos2) / sigma_next
                x_edm = x_hat + (sigma_next - sigma_hat) * (0.5 * dx + 0.5 * dx2)
                pos_edm = pos_hat + (sigma_next - sigma_hat) * (0.5 * dpos + 0.5 * dpos2)

        pos_edm = subtract_mean(pos_edm, batch)
        samples = []
        for i in range(num_molecules):
            mask = batch == i
            atom_types = x_edm[mask, :5].argmax(dim=-1)
            if self.config.dataset.use_charges:
                charges = (x_edm[mask, -1] * self.config.diffusion.normalize_charge_factor).round().long()
                samples.append((pos_edm[mask], atom_types, charges))
            else:
                samples.append((pos_edm[mask], atom_types))
        return samples

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.config.training.validation_frequency:
            return
        self._evaluate_samples(10000, "val")

    def test_step(self, batch, batch_idx):
        return None

    def on_test_epoch_end(self):
        self._evaluate_samples(10000, "test")

    def _evaluate_samples(self, count, phase):
        molecules = []
        for start in range(0, count, self.config.training.batch_size):
            molecules.extend(self.sample(min(self.config.training.batch_size, count - start)))
        atoms = mols = total_atoms = total_mols = 0
        for molecule in molecules:
            stable, stable_atoms, atom_count = check_stability(*molecule)
            atoms += stable_atoms
            total_atoms += atom_count
            mols += int(stable)
            total_mols += 1
        results = {
            f"{phase}_edm/atom_stability": 100 * atoms / max(1, total_atoms),
            f"{phase}_edm/molecule_stability": 100 * mols / max(1, total_mols),
        }
        if self.edm_analyzer:
            (validity, uniqueness, novelty), _ = self.edm_analyzer.evaluate(molecules)
            results.update({f"{phase}_edm/validity": validity,
                            f"{phase}_edm/uniqueness": uniqueness,
                            f"{phase}_edm/novelty": novelty})
        for key, value in results.items():
            self.log(key, value, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self, self.config.optimizer.weight_decay),
            lr=self.config.optimizer.lr,
        )
        if self.config.scheduler.use_cosine:
            T_max = self.config.scheduler.get("T_max_epochs", None) or self.trainer.max_epochs
            return {"optimizer": optimizer, "lr_scheduler": CosineAnnealingLR(
                optimizer, T_max=T_max,
                eta_min=self.config.optimizer.lr / 100,
            )}
        return optimizer


def main(config):
    print_config(config, "QM9 Flow Matching Configuration")
    pl.seed_everything(config.seed)
    train_loader, val_loader, sampler, edm_smiles, zatom_smiles, dataset_info = load_data(config)
    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    logger = None
    if config.logging.enabled:
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name + "-FM",
            entity=config.logging.get("wandb_identity", None),
            config=config.to_dict(),
            save_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs"),
        )
    callbacks = [pl.callbacks.ModelCheckpoint(
        monitor="val_edm/molecule_stability", mode="max",
        every_n_epochs=config.training.validation_frequency, save_last=True,
    ), TimerCallback()]
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(decay=config.training.get("ema_decay", 0.999),
                                     warmup_steps=config.training.get("ema_warmup_steps", 2000)))
    trainer = pl.Trainer(
        logger=logger, max_epochs=config.training.epochs, callbacks=callbacks,
        accelerator=accelerator, devices=config.system.gpus if accelerator == "gpu" else "auto",
        precision=config.system.precision, gradient_clip_val=config.training.gradient_clip_val,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
    )
    model = QM9FMModel(config)
    model.set_num_atoms_sampler(sampler)
    model.init_molecule_analyzer(dataset_info, edm_smiles, zatom_smiles)
    test_ckpt = config.testing.test_ckpt
    if test_ckpt:
        test_results = trainer.test(model, val_loader, ckpt_path=test_ckpt)
        save_test_metrics(config, test_results)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        trainer.test(model, val_loader, ckpt_path="last")


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/qm9_gen.yaml")
    args, unknown_args = parser.parse_known_args()
    main(load_with_defaults(dataset_config=args.config, cli_args=unknown_args))