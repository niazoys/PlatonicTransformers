import gc
import os
import sys
from typing import Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data

from platonic_transformers.datasets.omol import get_omol_loaders
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
)
from platonic_transformers.utils.utils import CosineWarmupScheduler, RandomSOd
from platonic_transformers.utils.callbacks import MemoryMonitorCallback, TimerCallback

# Performance optimizations
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class OMolModel(pl.LightningModule):
    """Lightning module for OMol energy and force prediction."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels_scalar = (
            92  # base atom features onehot
            + 3 * ("coords" in self.config.dataset.scalar_features)  # x,y,z coordinates as scalars
            + 1 * ("charges" in self.config.dataset.scalar_features)  # charges as scalars
        )
        in_channels_vector = 0  # No vector features used in this setup

        # --- Dynamically configure model outputs based on force prediction mode ---
        if self.config.model.predict_forces:
            # Direct force prediction: 1 scalar (energy) and 1 vector (force)
            out_channels_scalar = 1
            out_channels_vec = 1
            scalar_task_level = "graph"
            vector_task_level = "node"
        else:
            # Energy prediction only (forces from gradient)
            out_channels_scalar = 1
            out_channels_vec = 0
            scalar_task_level = "graph"
            vector_task_level = "graph"  # Not used for output, but required
        # --- End of dynamic configuration ---

        # Model specification
        solid_name = self.config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}")

        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=out_channels_scalar,
            output_dim_vec=out_channels_vec,
            nhead=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=3,
            dense_mode=self.config.model.dense_mode,
            scalar_task_level=scalar_task_level,
            vector_task_level=vector_task_level,
            ffn_readout=self.config.model.ffn_readout,
            mean_aggregation=self.config.model.mean_aggregation,
            dropout=self.config.model.dropout,
            drop_path_rate=self.config.model.drop_path_rate,
            layer_scale_init_value=self.config.model.layer_scale_init_value,
            attention=self.config.model.attention,
            ffn_dim_factor=4,
            rope_sigma=self.config.model.rope_sigma,
            ape_sigma=self.config.model.ape_sigma,
            learned_freqs=self.config.model.learned_freqs,
            freq_init=self.config.model.freq_init,
            use_key=self.config.model.use_key,
        )

        # Initialize normalization parameters
        self.register_buffer('shift', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('avg_num_nodes', torch.tensor(1.0, dtype=torch.float32))
        
        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.train_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        
        self.test_metrics_energy = torchmetrics.MeanAbsoluteError()
        self.test_metrics_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy_per_atom = torchmetrics.MeanAbsoluteError()

    def forward(self, graph: Data) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        graph = graph.to(self.device)
        # Prepare input features
        x = [graph.x]
        if "coords" in self.config.dataset.scalar_features:
            x.append(graph.pos)
        if "charges" in self.config.dataset.scalar_features:
            x.append(graph.charges[:, None])
        
        x = torch.cat(x, dim=-1)
        
        # Forward pass
        pred_scalar, pred_vec = self.net(x, graph.pos, graph.batch, vec=None, avg_num_nodes=self.avg_num_nodes.to(graph.pos.device))
        
        pred_energy = pred_scalar.view(-1)
        
        if self.config.model.predict_forces:
            # Squeeze the middle dimension: [N, 1, 3] -> [N, 3]
            pred_force = pred_vec.squeeze(1)
            return pred_energy, pred_force
        return pred_energy

    def pred_energy_and_force(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted energy and forces, using autograd if needed."""
        if self.config.model.predict_forces:
            # Model directly outputs energy and forces
            pred_energy, pred_force = self(graph)
            return pred_energy, pred_force

        # Calculate forces from energy gradient (autograd)
        with torch.enable_grad():
            graph.pos = graph.pos.clone().requires_grad_(True)
            pred_energy = self(graph)
            sign = -1.0
            pred_force = sign * torch.autograd.grad(
                pred_energy,
                graph.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        if not self.training:
            pred_energy = pred_energy.detach()
            pred_force = pred_force.detach()

        return pred_energy, pred_force

    def training_step(self, graph: Data, batch_idx: int) -> torch.Tensor:
        if self.config.training.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            graph.forces = torch.einsum('bij,bj->bi', rot_per_sample, graph.forces)
        
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        
        # Loss calculation
        energy_loss = torch.mean((pred_energy - ((graph.energy - self.shift) / self.scale))**2)
        force_loss = torch.mean(torch.sqrt(torch.sum((pred_force - graph.forces / self.scale)**2, -1)))
        loss = energy_loss + self.config.training.lambda_F * force_loss

        # Logging metrics (converted to meV and meV/Å)
        pred_energy_mev = (pred_energy.detach() * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force.detach() * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000
        
        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.train_metric(pred_energy_mev, true_energy_mev)
        self.train_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.train_metric_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)
        
        self.log("train MAE (energy) [meV]", self.train_metric, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train MAE (force) [meV/Å]", self.train_metric_force, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train MAE (energy/atom) [meV]", self.train_metric_energy_per_atom, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx % 250 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
        return loss

    def on_train_epoch_end(self) -> None:
        pass
    
    def validation_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        
        pred_energy_mev = (pred_energy * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000
        
        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms
        
        self.valid_metric(pred_energy_mev, true_energy_mev)
        self.valid_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.valid_metric_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)

    def on_validation_epoch_end(self) -> None:
        self.log("valid MAE (energy) [meV]", self.valid_metric, prog_bar=True, sync_dist=True)
        self.log("valid MAE (force) [meV/Å]", self.valid_metric_force, prog_bar=True, sync_dist=True)
        self.log("valid MAE (energy/atom) [meV]", self.valid_metric_energy_per_atom, prog_bar=True, sync_dist=True)
    
    def test_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        
        pred_energy_mev = (pred_energy * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000
        
        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms
        
        self.test_metrics_energy(pred_energy_mev, true_energy_mev)
        self.test_metrics_force(pred_force_mev_ang, true_force_mev_ang)
        self.test_metrics_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)

    def on_test_epoch_end(self) -> None:
        self.log("test MAE (energy) [meV]", self.test_metrics_energy, prog_bar=True, sync_dist=True)
        self.log("test MAE (force) [meV/Å]", self.test_metrics_force, prog_bar=True, sync_dist=True)
        self.log("test MAE (energy/atom) [meV]", self.test_metrics_energy_per_atom, prog_bar=True, sync_dist=True)
  
    def configure_optimizers(self) -> dict[str, object]:
        """Create optimizer and optional scheduler with custom decay groups."""

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, module in self.named_modules():
            for pn, param in module.named_parameters():
                full_name = f"{mn}.{pn}" if mn else pn

                if pn == 'freqs':
                    no_decay.add(full_name)
                elif pn.endswith('bias') or ('layer_scale' in pn):
                    no_decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif pn.endswith('kernel'):
                    decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        param_dict = {pn: param for pn, param in self.named_parameters() if param.requires_grad}
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            print(f"Warning: Parameters {missing_params} were not explicitly assigned. Adding to no_decay.")
            no_decay.update(missing_params)

        assert len(decay & no_decay) == 0, f"Parameters in both decay and no_decay sets: {decay & no_decay}"
        
        optim_groups = [
            {
                "params": [param_dict[name] for name in sorted(decay) if name in param_dict],
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {
                "params": [param_dict[name] for name in sorted(no_decay) if name in param_dict],
                "weight_decay": 0.0,
            },
        ]
        
        optim_groups = [group for group in optim_groups if group["params"]]

        optimizer = torch.optim.Adam(optim_groups, lr=self.config.optimizer.lr)
        if self.config.scheduler.use_cosine:
            scheduler = CosineWarmupScheduler(optimizer, self.config.scheduler.warmup_epochs, self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid MAE (energy) [meV]"}
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE (energy) [meV]"}


def main(config: ml_collections.ConfigDict) -> None:
    """Train and evaluate the Platonic Transformer on the OMol dataset."""
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader, _, _ = get_omol_loaders(
        root=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        num_workers=2 if config.system.gpus > 1 else config.system.num_workers,
        use_charges=False,
        seed=config.seed,
        debug_subset=config.dataset.debug_subset,
        referencing=config.dataset.referencing,
        include_hof=config.dataset.include_hof,
        scale_shift=config.dataset.scale_shift,
        recalculate=config.dataset.recalculate_stats,
        use_k_hot=config.dataset.use_khot_encoding,
    )

    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = config.system.gpus if accelerator == "gpu" else "auto"
        
    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(project="Platonic-omol", config=config.to_dict(), save_dir=save_dir)
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy) [meV]', mode='min', filename='best-energy-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (force) [meV/Å]', mode='min', filename='best-force-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy/atom) [meV]', mode='min', filename='best-energy-per-atom-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(save_last=True, filename='last'),
        TimerCallback(),
        MemoryMonitorCallback(log_frequency=50)
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if config.system.timer:
        callbacks.append(Timer(duration=config.system.timer))
   
    if config.testing.load_weights:
        model = OMolModel.load_from_checkpoint(checkpoint_path=config.testing.load_weights, config=config)
    else:
        model = OMolModel(config)

    if hasattr(train_loader.dataset, 'scale'):
        model.scale = torch.tensor(train_loader.dataset.scale).to(model.device)
        model.shift = torch.tensor(train_loader.dataset.shift).to(model.device)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
        inference_mode=False,
        strategy=DDPStrategy(find_unused_parameters=True) if config.system.gpus > 1 else 'auto'
    )

    if not config.testing.test_ckpt:
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        best_ckpt_path = callbacks[2].best_model_path or "last"
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
    else:
        model = OMolModel.load_from_checkpoint(
            config.testing.test_ckpt,
            hparams_file=os.path.join(os.path.dirname(config.testing.test_ckpt), "hparams.yaml"),
            config=config,
        )
        trainer.test(model, test_loader)

if __name__ == "__main__":
    # Parse command-line arguments (allow unknown for simple overrides)
    parser = get_arg_parser(default_config_path="configs/omol.yaml")
    args, unknown_args = parser.parse_known_args()
    
    # Load configuration and parse CLI overrides automatically
    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args  # Automatically infers parameter locations
    )
    
    # Run training
    main(config)
