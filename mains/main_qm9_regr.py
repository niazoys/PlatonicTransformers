import os
import sys
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import Timer
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from datasets.k_hot_encoding import KHOT_EMBEDDINGS
from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS
from models.platoformer.utils import scatter_add
from utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config
)
from utils.utils import CosineWarmupScheduler, RandomSOd
from utils.callbacks import StopOnPersistentDivergence, TimerCallback

# Performance optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class QM9Model(pl.LightningModule):
    """Lightning module for QM9 molecular property prediction using PlatonicTransformer."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config
        
        # Initialize k-hot embedding tensor if needed
        if config.dataset.use_k_hot_encoding:
            self._init_khot_embeddings()
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Standard input dimension from QM9 graph.x (atom type embeddings)
        # For QM9, graph.x typically has 11 features.
        input_feature_dimensionality = 97 if config.dataset.use_k_hot_encoding else 11

        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}")

        self.net = PlatonicTransformer(
            # Basic/essential specification:
            input_dim=input_feature_dimensionality,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=1,  # Single target property prediction
            output_dim_vec=0,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            # Pooling and readout specification:
            scalar_task_level=config.model.scalar_task_level,
            vector_task_level=config.model.vector_task_level,
            ffn_readout=config.model.ffn_readout,
            # Attention block specification:
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            # RoPE and APE specification:
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
        )
        # self.net = torch.compile(self.net)

        # Initialize normalization parameters
        self.register_buffer('shift', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))

        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

    def _init_khot_embeddings(self) -> None:
        """Initialize k-hot embedding tensor for fast lookup."""
        # QM9 has atoms with atomic numbers 1, 6, 7, 8, 9 (H, C, N, O, F)
        embedding_dim = len(next(iter(KHOT_EMBEDDINGS.values())))
        embedding_tensor = torch.zeros(5, embedding_dim, dtype=torch.float32)
        qm9_to_atomic = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
        for qm9_idx, atomic_num in qm9_to_atomic.items():
            if atomic_num in KHOT_EMBEDDINGS:
                embedding_tensor[qm9_idx] = torch.tensor(KHOT_EMBEDDINGS[atomic_num])
        self.register_buffer('khot_embedding_tensor', embedding_tensor)

    def include_k_hot_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.dataset.use_k_hot_encoding:
            return x
        atom_onehot = x[:, :5]
        atom_indices = torch.argmax(atom_onehot, dim=-1)
        embeddings = self.khot_embedding_tensor[atom_indices]
        return torch.cat([embeddings, x[:, -5:]], dim=-1)

    def forward(self, graph: Data) -> torch.Tensor:
        # Use the class method instead of the global function
        node_features = self.include_k_hot_encoding(graph.x)

        positions = graph.pos  # [N, 3]
        positions_mean = scatter_add(
            graph.pos,
            graph.batch,
            dim_size=graph.batch.max().item() + 1,
        ) / scatter_add(
            torch.ones_like(graph.pos[:, :1]),
            graph.batch,
            dim_size=graph.batch.max().item() + 1,
        )
        positions = positions - positions_mean[graph.batch]  # Center positions per graph
        batch_idx = graph.batch  # [N]

        # PlatonicTransformer expects x, pos, batch
        pred, _ = self.net(node_features, positions, batch_idx, vec=None, avg_num_nodes=self.avg_num_nodes)

        return pred.squeeze(-1)  # Assuming output_dim is 1
    
    def set_dataset_statistics(self, dataloader: DataLoader) -> None:
        """
        Compute and cache or load the mean and standard deviation of the target property.
        
        The statistics are saved to a file named 'stats_{target_name}.npz' in the
        dataset's root directory to avoid re-computation on subsequent runs.
        """
        stats_file = os.path.join(self.config.dataset.data_dir, f"stats_{self.config.dataset.target}.npz")

        if os.path.exists(stats_file):
            print(f"Loading dataset statistics from cached file: {stats_file}")
            stats = np.load(stats_file)
            self.shift = torch.tensor(stats['shift'])
            self.scale = torch.tensor(stats['scale'])
            self.avg_num_nodes = torch.tensor(stats['avg_num_nodes'])
        else:
            print('Computing dataset statistics...')
            ys = []
            total_num_nodes = 0
            for data in dataloader:
                ys.append(data.y)
                total_num_nodes += data.num_nodes
            ys = np.concatenate(ys)
            
            self.shift = torch.tensor(np.mean(ys))
            self.scale = torch.tensor(np.std(ys))
            self.avg_num_nodes = torch.tensor(total_num_nodes / len(dataloader.dataset))

            print(f"Saving dataset statistics to {stats_file}")
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            np.savez(stats_file, shift=self.shift, scale=self.scale, avg_num_nodes=self.avg_num_nodes)

        print(f'Target statistics - Mean: {self.shift:.4f}, Std: {self.scale:.4f}')

    def training_step(self, graph: Data, batch_idx: int) -> torch.Tensor:
        # Apply rotation augmentation if enabled
        if self.config.training.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            
        pred = self(graph)
        loss = torch.mean(torch.abs(pred - (graph.y - self.shift) / self.scale))
        self.train_metric(pred * self.scale + self.shift, graph.y)
        return loss

    def validation_step(self, graph: Data, batch_idx: int) -> None:
        pred = self(graph)
        self.valid_metric(pred * self.scale + self.shift, graph.y)

    def test_step(self, graph: Data, batch_idx: int) -> None:
        pred = self(graph)
        self.test_metric(pred * self.scale + self.shift, graph.y)

    def on_train_epoch_end(self) -> None:
        self.log("train MAE", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("valid MAE", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log("test MAE", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self) -> dict[str, object]:
        """Configure optimizer with weight decay and learning rate schedule."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():  # mn is module name, m is module instance
            for pn, p in m.named_parameters():  # pn is parameter name (e.g., 'weight', 'bias', 'freqs')
                fpn = f'{mn}.{pn}' if mn else pn  # fpn is the full parameter name

                if pn == 'freqs':
                    no_decay.add(fpn)
                elif pn.endswith('bias') or ('layer_scale' in pn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('kernel'):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                # Parameters not matching any rule will be caught later and added to no_decay by default.

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        current_params_in_groups = decay | no_decay
        missing_params = param_dict.keys() - current_params_in_groups
        if missing_params:
            print(f"Warning: Parameters {missing_params} were not explicitly assigned to decay/no_decay by specific rules. Adding to no_decay by default.")
            no_decay.update(missing_params) # Add missing parameters to no_decay group

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} found in both decay and no_decay sets!"
        
        # Ensure all learnable parameters are covered
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not assigned to any optimizer group!"

        optim_groups = [
            {"params": [param_dict[p_name] for p_name in sorted(list(decay)) if p_name in param_dict], "weight_decay": self.config.optimizer.weight_decay},
            {"params": [param_dict[p_name] for p_name in sorted(list(no_decay)) if p_name in param_dict], "weight_decay": 0.0},
        ]
        
        # Filter out empty groups (e.g., if 'decay' set is empty)
        optim_groups = [group for group in optim_groups if group["params"]]

        if not optim_groups and list(param_dict.keys()): # Should not happen if there are learnable params
            raise ValueError("No optimizer groups were created, but there are learnable parameters.")
        elif not optim_groups and not list(param_dict.keys()): # No learnable params
             print("Warning: No learnable parameters found for the optimizer.")

        optimizer = torch.optim.Adam(optim_groups, lr=self.config.optimizer.lr)
        if self.config.scheduler.use_cosine:
            scheduler = CosineWarmupScheduler(optimizer, self.config.scheduler.warmup_epochs, self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid MAE"}
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE"}

def load_data(config: ml_collections.ConfigDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and preprocess QM9 dataset."""
    # Load dataset
    dataset = QM9(root=config.dataset.data_dir)
    
    # Select target property, with a preference for '_atom' versions if available.
    all_targets = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 
        'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
    ]
    target_map = {name: i for i, name in enumerate(all_targets)}

    target_name = config.dataset.target
    atom_version = f"{target_name}_atom"
    if atom_version in target_map:
        print(f"Redirecting target '{target_name}' to '{atom_version}'.")
        target_name = atom_version
        
    try:
        target_idx = target_map[target_name]
    except KeyError:
        raise ValueError(f"Target '{target_name}' not found in QM9 targets: {all_targets}")
    
    # Set the target for the entire dataset *before* splitting
    dataset.data.y = dataset.data.y[:, target_idx]

    # Create train/val/test split (same as DimeNet)
    random_state = np.random.RandomState(seed=42)
    perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
    train_idx, val_idx, test_idx = perm[:110000], perm[110000:120000], perm[120000:]
    datasets = {'train': dataset[train_idx], 'val': dataset[val_idx], 'test': dataset[test_idx]}
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            split_dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.system.num_workers,
        )
        for split, split_dataset in datasets.items()
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def main(config: ml_collections.ConfigDict) -> None:
    """Train and evaluate the Platonic Transformer on QM9."""
    print_config(config, "QM9 Training Configuration")
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader = load_data(config)

    if config.system.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.system.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
        
    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            config=config.to_dict(),
            save_dir=save_dir
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='valid MAE', mode='min', 
                                   every_n_epochs=1, save_last=True),
        TimerCallback()
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if config.system.timer is not None:
        callbacks.append(Timer(duration=config.system.timer))
    
    # Add early stopping callback if configured
    if config.callbacks.early_stopping.enabled:
        es_config = config.callbacks.early_stopping
        callbacks.append(StopOnPersistentDivergence(
            monitor=es_config.monitor,
            threshold=es_config.threshold,
            patience=es_config.patience,
            grace_epochs=es_config.grace_epochs,
            verbose=False
        ))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision
    )

    test_ckpt = config.testing.test_ckpt
    if test_ckpt is None:
        model = QM9Model(config)
        model.set_dataset_statistics(train_loader)
        resume_ckpt = config.testing.resume_ckpt
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
        # Test with the best checkpoint from training
        best_ckpt_path = callbacks[0].best_model_path if callbacks[0].best_model_path else "last"
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)

    else:
        # When loading from checkpoint
        model = QM9Model.load_from_checkpoint(test_ckpt)
        model.set_dataset_statistics(train_loader)  # Recompute stats or ensure they are loaded
        trainer.test(model, test_loader)


if __name__ == "__main__":
    # Parse command-line arguments (allow unknown for simple overrides)
    parser = get_arg_parser(default_config_path="configs/qm9_regr.yaml")
    args, unknown_args = parser.parse_known_args()
    
    # Load configuration and parse CLI overrides automatically
    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args  # Automatically infers parameter locations
    )
    
    # Run training
    main(config)
