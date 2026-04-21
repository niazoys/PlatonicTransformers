import os
import sys
import time
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Timer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from platonic_transformers.datasets.qm9 import QM9Dataset, collate_fn
from platonic_transformers.datasets.qm9_bond_analyze import check_stability
from platonic_transformers.datasets.qm9_rdkit_utils import BasicMolecularMetrics
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import TimerCallback
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.utils.utils import RandomSOd, fully_connected_edge_index, subtract_mean

# Performance optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


# ---------------------------------------------------------------------------
# EDM preconditioning, loss, sampler (Karras et al., 2022)
# ---------------------------------------------------------------------------


class EDMPrecond(torch.nn.Module):
    """Karras-style preconditioning wrapper for an equivariant denoiser."""

    def __init__(
        self,
        model: torch.nn.Module,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1.0,
        avg_num_nodes: float = 18.0,
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.avg_num_nodes = avg_num_nodes

    def forward(self, x, pos, batch, sigma):
        # sigma may be per-node (shape [N] or [N,1], e.g. from EDMLoss) or per-graph
        # (shape [B] or [B,1], e.g. from the sampler). Normalize to a per-graph vector.
        sigma = sigma.reshape(-1, 1)
        num_graphs = int(batch.max().item()) + 1
        if sigma.shape[0] == batch.shape[0]:
            # Per-node form — reduce to per-graph (values are identical within a graph).
            sigma_per_graph = torch.unique_consecutive(sigma.squeeze(-1)).reshape(-1, 1)
        elif sigma.numel() == 1:
            sigma_per_graph = sigma.expand(num_graphs, 1)
        else:
            sigma_per_graph = sigma
        sigma_per_node = sigma_per_graph[batch]  # (N, 1)

        c_skip = self.sigma_data ** 2 / (sigma_per_node ** 2 + self.sigma_data ** 2)
        c_out = sigma_per_node * self.sigma_data / (sigma_per_node ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_per_node ** 2).sqrt()
        c_noise = sigma_per_graph.log() / 4  # (B, 1)

        x_in = c_in * x
        pos_in = c_in * pos

        # Also feed noise as a per-node scalar input feature (matches cleaned-dev).
        scalars_in = torch.cat([x_in, c_noise[batch]], dim=-1)

        scalars_out, vecs_out = self.model(
            scalars_in, pos_in, batch, vec=None,
            t=c_noise.squeeze(-1), avg_num_nodes=self.avg_num_nodes,
        )
        dx = scalars_out
        dpos = vecs_out.squeeze(1)

        F_x = x_in - dx
        F_pos = pos_in - dpos
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_pos = c_skip * pos + c_out * F_pos.to(torch.float32)
        return D_x, D_pos


class EDMLoss:
    """Karras-style log-normal noise + weighted MSE on atom features and positions."""

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 1.0,
        normalize_x_factor: float = 4.0,
        normalize_charge_factor: float = 8.0,
        use_charges: bool = True,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.normalize_charge_factor = normalize_charge_factor
        self.use_charges = use_charges

    def __call__(self, net: EDMPrecond, inputs: dict):
        pos, x, batch = inputs["pos"], inputs["x"], inputs["batch"]
        pos = subtract_mean(pos, batch)

        if self.use_charges:
            x = x.clone()
            x[:, :-1] = x[:, :-1] / self.normalize_x_factor
            x[:, -1] = x[:, -1] / self.normalize_charge_factor
        else:
            x = x / self.normalize_x_factor

        rnd_normal = torch.randn(
            [batch.max() + 1, 1], device=pos.device, dtype=torch.float32
        )
        rnd_normal = rnd_normal[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        x_noisy = x + torch.randn_like(x) * sigma
        pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma

        D_x, D_pos = net(x_noisy, pos_noisy, batch, sigma)
        error_x = (D_x - x) ** 2
        error_pos = (D_pos - pos) ** 2
        loss = (weight * error_x).mean() + (weight * error_pos).mean()
        return loss, (D_x, D_pos)


def edm_sampler(
    net: EDMPrecond,
    pos_0: torch.Tensor,
    x_0: torch.Tensor,
    batch: torch.Tensor,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 20.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Karras Euler-Maruyama sampler with second-order correction."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    num_graphs = int(batch.max().item()) + 1

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=pos_0.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, pos_cur = x_next, pos_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        pos_hat = pos_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(pos_cur)

        x_denoised, pos_denoised = net(x_hat, pos_hat, batch, t_hat.expand(num_graphs))
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        if i < num_steps - 1:
            x_denoised, pos_denoised = net(x_next, pos_next, batch, t_next.expand(num_graphs))
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

    pos_next = subtract_mean(pos_next, batch)
    return x_next, pos_next


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class QM9GenModel(pl.LightningModule):
    """Lightning module for equivariant diffusion generation on QM9."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})
        self.config = config

        use_charges = config.dataset.use_charges
        in_scalar = 5 + (1 if use_charges else 0) + 1  # atom types + optional charge + noise feature
        out_scalar = 5 + (1 if use_charges else 0)

        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}"
            )

        self.net = PlatonicTransformer(
            input_dim=in_scalar,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=out_scalar,
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
            rope_on_values=config.model.rope_on_values,
            time_conditioning=True,
        )

        if getattr(config.model, "compile", True):
            self.net = torch.compile(self.net)

        self.model = EDMPrecond(
            self.net,
            sigma_data=config.diffusion.sigma_data,
            avg_num_nodes=config.diffusion.avg_num_nodes,
        )
        self.criterion = EDMLoss(
            P_mean=config.diffusion.P_mean,
            P_std=config.diffusion.P_std,
            sigma_data=config.diffusion.sigma_data,
            normalize_x_factor=config.diffusion.normalize_x_factor,
            normalize_charge_factor=config.diffusion.normalize_charge_factor,
            use_charges=use_charges,
        )
        self.rotation_generator = RandomSOd(3)

        self.num_atoms_sampler = None
        self.molecule_analyzer = None

    def set_num_atoms_sampler(self, num_atoms_sampler):
        self.num_atoms_sampler = num_atoms_sampler

    def init_molecule_analyzer(self, dataset_info, smiles_list):
        self.molecule_analyzer = BasicMolecularMetrics(dataset_info, smiles_list)

    def training_step(self, batch, batch_idx):
        if self.config.training.train_augm:
            rot = self.rotation_generator().type_as(batch["pos"])
            batch["pos"] = torch.einsum("ij,bj->bi", rot, batch["pos"])
        loss, _ = self.criterion(self.model, batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch["batch"].max() + 1,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.criterion(self.model, batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch["batch"].max() + 1,
        )
        return loss

    def on_validation_epoch_end(self):
        freq = self.config.training.validation_frequency
        full_validation = (self.current_epoch + 1) % freq == 0

        results = self.run_validation_sampling(
            num_molecules=10000 if full_validation else self.config.training.batch_size,
            batch_size=self.config.training.batch_size,
            rdkit_metrics=full_validation,
        )
        if not full_validation:
            results = {f"{k} (estimate)": v for k, v in results.items()}
        for key, value in results.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_test_epoch_end(self):
        results = self.run_validation_sampling(
            num_molecules=10000,
            batch_size=self.config.training.batch_size,
            rdkit_metrics=True,
        )
        for key, value in results.items():
            self.log(f"final/{key}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        if self.config.scheduler.use_cosine:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.config.optimizer.lr / 100,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    # -----------------------------------------------------------------------
    # Sampling / evaluation
    # -----------------------------------------------------------------------

    def sample(self, num_molecules: int):
        """Generate a batch of molecules and return [(pos, atom_types[, charges]), ...]."""
        if self.num_atoms_sampler is None:
            raise RuntimeError("num_atoms_sampler not initialized; call set_num_atoms_sampler().")

        self.eval()
        use_charges = self.config.dataset.use_charges
        with torch.no_grad():
            num_atoms = self.num_atoms_sampler(num_molecules).to(self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)

            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            pos_0 = subtract_mean(pos_0, batch_idx)
            x_0 = torch.randn(
                [len(batch_idx), 5 + (1 if use_charges else 0)], device=self.device
            )

            x_final, pos_final = edm_sampler(
                self.model,
                pos_0,
                x_0,
                batch_idx,
                num_steps=self.config.diffusion.num_steps,
                sigma_min=self.config.diffusion.sigma_min,
                sigma_max=self.config.diffusion.sigma_max,
                rho=self.config.diffusion.rho,
                S_churn=self.config.diffusion.S_churn,
            )

        sample_list = []
        for i in range(batch_idx.max() + 1):
            positions = pos_final[batch_idx == i]
            if use_charges:
                atom_types = x_final[batch_idx == i, :-1].argmax(dim=-1)
                charges = (
                    x_final[batch_idx == i, -1] * self.config.diffusion.normalize_charge_factor
                ).round().long()
                sample_list.append((positions, atom_types, charges))
            else:
                atom_types = x_final[batch_idx == i].argmax(dim=-1)
                sample_list.append((positions, atom_types))
        return sample_list

    def run_validation_sampling(
        self, num_molecules: int = 10000, batch_size: int = 100, rdkit_metrics: bool = True
    ) -> dict:
        """Generate molecules and compute stability + (optionally) RDKit metrics."""
        if self.molecule_analyzer is None and rdkit_metrics:
            raise RuntimeError(
                "molecule_analyzer not initialized; call init_molecule_analyzer()."
            )

        steps = max(1, num_molecules // batch_size)
        molecules = []
        t0 = time.time()
        for _ in trange(steps, desc="sampling", leave=False):
            molecules += self.sample(batch_size)
        avg_time_per_molecule = (time.time() - t0) / max(1, len(molecules))

        # Stability
        count_atm_stable = count_atm_total = count_mol_stable = count_mol_total = 0
        for mol in molecules:
            is_stable, nr_stable, total = check_stability(*mol)
            count_atm_stable += nr_stable
            count_atm_total += total
            count_mol_stable += int(is_stable)
            count_mol_total += 1

        results = {
            "atom_stability": 100.0 * count_atm_stable / max(1, count_atm_total),
            "molecule_stability": 100.0 * count_mol_stable / max(1, count_mol_total),
            "avg_time_per_molecule": avg_time_per_molecule,
        }

        if rdkit_metrics:
            [validity, uniqueness, novelty], _ = self.molecule_analyzer.evaluate(molecules)
            results.update({
                "validity": validity,
                "uniqueness": uniqueness,
                "novelty": novelty,
                "discovery": validity * uniqueness * novelty,
            })
        return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(config: ml_collections.ConfigDict):
    train_set = QM9Dataset(
        split="train", root=config.dataset.data_dir, use_charges=config.dataset.use_charges
    )
    val_set = QM9Dataset(
        split="val", root=config.dataset.data_dir, use_charges=config.dataset.use_charges
    )

    num_atoms_sampler = train_set.NumAtomsSampler()
    smiles_list = [train_set[i]["smiles"] for i in range(len(train_set))]
    dataset_info = train_set.dataset_info

    if config.dataset.dataset_fraction < 1.0:
        subset_len = int(len(train_set) * config.dataset.dataset_fraction)
        train_set = torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[:subset_len]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "QM9 Generation Configuration")
    pl.seed_everything(config.seed)

    train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info = load_data(config)

    if config.system.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.system.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name, config=config.to_dict(), save_dir=save_dir
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="molecule_stability",
            mode="max",
            every_n_epochs=config.training.validation_frequency,
            save_last=True,
        ),
        TimerCallback(),
    ]
    if config.system.timer is not None:
        callbacks.append(Timer(duration=config.system.timer))
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    trainer_kwargs = dict(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
    )
    # Optional profiling / short-run knobs
    if config.training.get("max_steps", -1) > 0:
        trainer_kwargs["max_steps"] = config.training.max_steps
    if config.training.get("limit_train_batches", 1.0) != 1.0:
        trainer_kwargs["limit_train_batches"] = config.training.limit_train_batches
    if config.training.get("limit_val_batches", 1.0) != 1.0:
        trainer_kwargs["limit_val_batches"] = config.training.limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)

    test_ckpt = config.testing.test_ckpt
    if test_ckpt is None:
        model = QM9GenModel(config)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        best_ckpt = callbacks[0].best_model_path or "last"
        trainer.test(model, val_loader, ckpt_path=best_ckpt)
    else:
        model = QM9GenModel.load_from_checkpoint(test_ckpt)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
        trainer.test(model, val_loader)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/qm9_gen.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)
