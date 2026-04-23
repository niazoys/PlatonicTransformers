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
from rdkit import Chem  # used for PDB serialization when logging wandb.Molecule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from platonic_transformers.datasets.qm9 import QM9Dataset, collate_fn
from platonic_transformers.datasets.qm9_bond_analyze import check_stability
from platonic_transformers.datasets.qm9_rdkit_utils import (
    BasicMolecularMetrics,
    ZatomMolecularMetrics,
    compute_training_smiles,
    compute_training_smiles_zatom,
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
from platonic_transformers.utils.utils import RandomSOd, fully_connected_edge_index, subtract_mean

# Performance backends (mirrors our OMol training setup).
# Keep weights/activations in fp32 (diffusion loss weighting is sigma-sensitive and
# the weighted MSE on positions can be swamped by bf16 rounding at small sigma).
# Matmul precision is set per-run from config.system.float32_matmul_precision
# ("highest" = fp32 operands, "high" = TF32, "medium" = bf16 operands, fp32 accumulator).
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 128  # variable atom count per batch


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
    return_trajectory: bool = False,
):
    """Karras Euler-Maruyama sampler with second-order correction.

    If return_trajectory=True, additionally returns a list of
    (x, pos) snapshots at each sampler step (on CPU) for visualization.
    """
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
    trajectory = [(x_next.detach().cpu(), pos_next.detach().cpu())] if return_trajectory else None

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

        if return_trajectory:
            trajectory.append((x_next.detach().cpu(), pos_next.detach().cpu()))

    pos_next = subtract_mean(pos_next, batch)
    if return_trajectory:
        return x_next, pos_next, trajectory
    return x_next, pos_next


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class QM9GenModel(pl.LightningModule):
    """Lightning module for equivariant diffusion generation on QM9."""

    def __init__(self, config):
        super().__init__()
        # Lightning's load_from_checkpoint passes the saved hyper_parameters
        # back as plain Python dicts; accept either a ConfigDict or dict.
        if not isinstance(config, ml_collections.ConfigDict):
            config = ml_collections.ConfigDict(config)
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
            attention_backend=config.model.get("attention_backend", "scatter"),
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
        self.edm_analyzer = None
        self.zatom_analyzer = None

    def set_num_atoms_sampler(self, num_atoms_sampler):
        self.num_atoms_sampler = num_atoms_sampler

    def init_molecule_analyzer(self, dataset_info, edm_smiles_list, zatom_smiles_list):
        """Build both metric analyzers. See qm9_rdkit_utils module docstring
        for the two protocols (EDM and Zatom-1) and the clearly-documented
        deviation from Zatom-1 (we substitute RDKit's PDB writer for
        pymatgen+OpenBabel)."""
        self.edm_analyzer = BasicMolecularMetrics(dataset_info, edm_smiles_list)
        self.zatom_analyzer = ZatomMolecularMetrics(dataset_info, zatom_smiles_list)

    def training_step(self, batch, batch_idx):
        if self.config.training.train_augm:
            rot = self.rotation_generator().type_as(batch["pos"])
            batch["pos"] = torch.einsum("ij,bj->bi", rot, batch["pos"])
        loss, _ = self.criterion(self.model, batch)
        self.log(
            "loss/train",
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
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch["batch"].max() + 1,
        )
        return loss

    def on_validation_epoch_end(self):
        # Only run the expensive sample-and-evaluate pass on "full" validation
        # epochs (every validation_frequency epochs). The per-epoch val loss is
        # already tracked by validation_step.
        freq = self.config.training.validation_frequency
        full_validation = (self.current_epoch + 1) % freq == 0
        if not full_validation:
            return

        results = self.run_validation_sampling(
            num_molecules=10000,
            batch_size=self.config.training.batch_size,
            rdkit_metrics=True,
            posebusters_max_molecules=1000,
        )
        for key, value in results.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_test_epoch_end(self):
        # Final test pass: match Zatom-1 Table 2 protocol — no sub-sampling
        # anywhere, even for PoseBusters. Takes a bit longer (PoseBusters on
        # ~9k valid mols) but gives apples-to-apples numbers for the paper
        # table.
        results = self.run_validation_sampling(
            num_molecules=10000,
            batch_size=self.config.training.batch_size,
            rdkit_metrics=True,
            posebusters_max_molecules=-1,
        )
        for key, value in results.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
        self, num_molecules: int = 10000, batch_size: int = 100,
        rdkit_metrics: bool = True, posebusters_max_molecules: int = 1000,
    ) -> dict:
        """Generate molecules and compute stability (always) + RDKit/PoseBusters
        metrics (only when rdkit_metrics=True).

        Returns a flat dict of scalars. When rdkit_metrics is True we compute
        two parallel protocols in the same pass:
          - `*_edm`   : EDM (Hoogeboom et al. 2022) distance-table bonds with H.
          - `*_zatom` : Zatom-1 (arXiv:2602.22251) PDB-roundtrip bonds, no H.
        plus `posebusters_pass_rate` and per-check PoseBusters pass rates
        reported by the Zatom-1 paper (run on the sanitized Zatom-style mols).
        """
        if rdkit_metrics and (self.edm_analyzer is None or self.zatom_analyzer is None):
            raise RuntimeError(
                "Molecular analyzers not initialized; call init_molecule_analyzer()."
            )

        steps = max(1, num_molecules // batch_size)
        molecules = []
        t0 = time.time()
        for _ in trange(steps, desc="sampling", leave=False):
            molecules += self.sample(batch_size)
        avg_time_per_molecule = (time.time() - t0) / max(1, len(molecules))

        # Stability (EDM, always computed — cheap and the headline metric)
        count_atm_stable = count_atm_total = count_mol_stable = count_mol_total = 0
        for mol in molecules:
            is_stable, nr_stable, total = check_stability(*mol)
            count_atm_stable += nr_stable
            count_atm_total += total
            count_mol_stable += int(is_stable)
            count_mol_total += 1

        results = {
            "stability/atom": 100.0 * count_atm_stable / max(1, count_atm_total),
            "stability/molecule": 100.0 * count_mol_stable / max(1, count_mol_total),
            "timing/avg_per_molecule": avg_time_per_molecule,
        }

        if rdkit_metrics:
            # EDM-flavor validity/uniqueness/novelty
            [v_e, u_e, n_e], _ = self.edm_analyzer.evaluate(molecules)
            results.update({
                "validity/edm": v_e,
                "uniqueness/edm": u_e,
                "novelty/edm": n_e,
            })

            # Zatom-1-flavor validity/uniqueness/novelty + PoseBusters
            z = self.zatom_analyzer.evaluate(molecules)
            results.update({
                "validity/zatom": z["validity"],
                "uniqueness/zatom": z["uniqueness"],
                "novelty/zatom": z["novelty"],
            })
            results.update(run_posebusters(z["valid_mols"], max_molecules=posebusters_max_molecules))

            # Log a handful of sampled 3D molecules to wandb so we can watch
            # the qualitative output evolve. Each log entry is a PDB string
            # (~1-2 KB) so even ~20 full validations x 8 mols = ~160 KB total.
            self._log_sample_molecules(z["valid_mols"], n=8)

            # Log denoising trajectories for a few molecules — a multi-MODEL
            # PDB file is rendered by wandb's 3dmol.js viewer as a step-slider,
            # letting us scrub from pure noise (step 0) to the final sample
            # (step N). This is a separate extra call to self.sample(); adds
            # about one extra batch of sampling work per full validation.
            self._log_denoising_animations(n_molecules=4, n_frames=26)
        return results

    def _log_sample_molecules(self, valid_mols, n: int = 8) -> None:
        """Log up to n 3D molecule samples to wandb as an interactive viewer.

        Silent no-op when either (a) the active logger is not a WandB logger,
        or (b) wandb / Chem.MolToPDBBlock misbehave — visualization should
        never block training.
        """
        if not valid_mols:
            return
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        # WandB runs expose `.log` with a dict accepting wandb.Molecule entries;
        # non-WandB experiment objects may not support this so we guard the import.
        try:
            import wandb
            if not hasattr(experiment, "log"):
                return
            mols_to_log = valid_mols[:n]
            wandb_mols = []
            for m in mols_to_log:
                pdb = Chem.MolToPDBBlock(m)
                # wandb.Molecule.from_rdkit is new; fall back to PDB text via a
                # temporary file which every wandb version accepts.
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".pdb", delete=False
                ) as fh:
                    fh.write(pdb)
                    path = fh.name
                wandb_mols.append(wandb.Molecule(path))
            experiment.log({
                "samples/zatom_valid_molecules": wandb_mols,
                "samples/epoch": int(self.current_epoch),
            })
        except Exception as e:
            # Don't let logging kill training.
            print(f"[warn] wandb molecule logging skipped: {e}")

    def _log_denoising_animations(self, n_molecules: int = 4, n_frames: int = 26) -> None:
        """Log denoising trajectories (pure noise -> final sample) as multi-MODEL
        PDB files for wandb's 3dmol.js viewer to render as an animation.

        Runs `self.sample(n_molecules, return_trajectory=True)` and slices the
        trajectory to `n_frames` evenly-spaced snapshots. Each snapshot becomes
        a MODEL in a single PDB file per molecule; 3dmol.js plays MODEL 1..N as
        frames with a step slider.

        Silent no-op on non-WandB loggers or if anything goes wrong.
        """
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment

        try:
            import tempfile
            import wandb
            if not hasattr(experiment, "log"):
                return

            use_charges = self.config.dataset.use_charges
            atom_decoder = self.zatom_analyzer.dataset_info["atom_decoder"] \
                if self.zatom_analyzer else ["H", "C", "N", "O", "F"]

            # Generate a small batch with trajectories.
            trajectories = self._sample_with_trajectory(n_molecules)
            if not trajectories:
                return

            # For each molecule, emit a multi-MODEL PDB assembled from the
            # selected trajectory frames.
            wandb_mols = []
            for traj in trajectories:
                # Evenly-spaced frames across the full trajectory.
                total = len(traj["frames"])
                step_idxs = np.linspace(0, total - 1, num=min(n_frames, total)).round().astype(int)
                pdb_lines = []
                for model_num, t_idx in enumerate(step_idxs, start=1):
                    x_frame, pos_frame = traj["frames"][t_idx]
                    atom_types = x_frame[:, :5].argmax(dim=-1) if x_frame.shape[1] >= 5 else x_frame.argmax(dim=-1)
                    pdb_lines.append(f"MODEL {model_num}")
                    for i, (sym_idx, (xi, yi, zi)) in enumerate(zip(atom_types.tolist(), pos_frame.tolist()), start=1):
                        sym = atom_decoder[int(sym_idx)]
                        # Right-padded element to 2 chars for PDB column rules.
                        pdb_lines.append(
                            f"HETATM{i:>5d} {sym:<3s}  MOL     1    "
                            f"{xi:>8.3f}{yi:>8.3f}{zi:>8.3f}  1.00  0.00          {sym:>2s}"
                        )
                    pdb_lines.append("ENDMDL")
                pdb_lines.append("END")
                pdb_text = "\n".join(pdb_lines) + "\n"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
                    fh.write(pdb_text)
                    wandb_mols.append(wandb.Molecule(fh.name))

            experiment.log({
                "samples/denoising_animation": wandb_mols,
                "samples/denoising_animation_epoch": int(self.current_epoch),
            })
        except Exception as e:
            print(f"[warn] wandb denoising-animation logging skipped: {e}")

    def _sample_with_trajectory(self, num_molecules: int) -> list:
        """Generate `num_molecules` samples and keep the full denoising trajectory
        of each. Returns list of dicts: {"batch_mask", "frames"} where frames is
        a list of (x, pos) snapshots across sampler steps (one per step+1)."""
        if self.num_atoms_sampler is None:
            return []
        use_charges = self.config.dataset.use_charges
        self.eval()
        with torch.no_grad():
            num_atoms = self.num_atoms_sampler(num_molecules).to(self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)
            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            pos_0 = subtract_mean(pos_0, batch_idx)
            x_0 = torch.randn(
                [len(batch_idx), 5 + (1 if use_charges else 0)], device=self.device
            )
            out = edm_sampler(
                self.model,
                pos_0,
                x_0,
                batch_idx,
                num_steps=self.config.diffusion.num_steps,
                sigma_min=self.config.diffusion.sigma_min,
                sigma_max=self.config.diffusion.sigma_max,
                rho=self.config.diffusion.rho,
                S_churn=self.config.diffusion.S_churn,
                return_trajectory=True,
            )
            _x_final, _pos_final, trajectory = out

        # Re-slice trajectory per-molecule.
        batch_cpu = batch_idx.cpu()
        trajectories = []
        for g in range(int(batch_cpu.max()) + 1):
            mask = (batch_cpu == g)
            frames = []
            for (x_step, pos_step) in trajectory:
                frames.append((x_step[mask], pos_step[mask]))
            trajectories.append({"batch_mask": mask, "frames": frames})
        return trajectories


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
    dataset_info = train_set.dataset_info
    # Training-set SMILES for novelty must come through the same bond-inference
    # pipeline that generated molecules go through, otherwise protonation /
    # aromaticity differences between the SDF-derived SMILES and the evaluator
    # SMILES produce spurious novelty. We build one reference set per protocol:
    #   * EDM     : distance-table bonds, keep H, non-isomeric canonical SMILES.
    #   * Zatom-1 : PDB-roundtrip bonds, removeHs=True, isomeric SMILES.
    # Both are cached on disk so this only runs once per dataset.
    edm_cache = os.path.join(config.dataset.data_dir, "train_smiles_edm.pkl")
    # Zatom-1 default is removeHs=False so the training SMILES must retain H
    # through the PDB round-trip pipeline for the novelty reference. We use a
    # distinct cache filename from the earlier (removeHs=True) cache to avoid
    # loading stale data.
    zatom_cache = os.path.join(config.dataset.data_dir, "train_smiles_zatom_keep_hs.pkl")
    edm_smiles_list = compute_training_smiles(train_set, dataset_info, cache_path=edm_cache)
    zatom_smiles_list = compute_training_smiles_zatom(train_set, dataset_info, cache_path=zatom_cache)

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

    return (
        train_loader,
        val_loader,
        num_atoms_sampler,
        edm_smiles_list,
        zatom_smiles_list,
        dataset_info,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "QM9 Generation Configuration")
    torch.set_float32_matmul_precision(config.system.get("float32_matmul_precision", "high"))
    pl.seed_everything(config.seed)

    (
        train_loader,
        val_loader,
        num_atoms_sampler,
        edm_smiles_list,
        zatom_smiles_list,
        dataset_info,
    ) = load_data(config)

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
            monitor="stability/molecule",
            mode="max",
            every_n_epochs=config.training.validation_frequency,
            save_last=True,
        ),
        TimerCallback(),
    ]
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(
            decay=config.training.get("ema_decay", 0.9999),
            warmup_steps=config.training.get("ema_warmup_steps", 2000),
        ))
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
        model.init_molecule_analyzer(dataset_info, edm_smiles_list, zatom_smiles_list)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        # Use the LAST checkpoint, not best — with EMA, the last-epoch EMA
        # weights are a tighter estimate of the model's asymptotic quality
        # than picking whichever checkpoint happened to score highest on its
        # single 10k-molecule eval (which is dominated by stochastic noise).
        trainer.test(model, val_loader, ckpt_path="last")
    else:
        # Test-only path. Instantiate the model from the current config, then
        # let `trainer.test(ckpt_path=...)` handle full checkpoint restoration
        # (model weights + callback states including EMA). This is the only
        # API that properly triggers the EMA callback's load_state_dict hook;
        # `LightningModule.load_from_checkpoint` restores weights alone.
        model = QM9GenModel(config)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, edm_smiles_list, zatom_smiles_list)
        trainer.test(model, val_loader, ckpt_path=test_ckpt)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/qm9_gen.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)
