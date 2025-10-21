import os
import ssl
import sys
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

# Import optimizer
try:
    from timm.optim import Lamb
except ImportError:
    print("Warning: timm not installed. LAMB optimizer will not be available.")
    Lamb = None

# Import model and utilities
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config
)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.utils.utils import CosineWarmupScheduler, RandomSOd
from platonic_transformers.utils.callbacks import TimerCallback

# Allow CIFAR-10 download on servers with SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Performance optimization
torch.set_float32_matmul_precision('medium')


class CIFAR10Model(pl.LightningModule):
    """LightningModule implementing CIFAR-10 classification with config-based setup."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        # Save hyperparameters (ml_collections ConfigDict is serializable)
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config
        
        # Setup 2D rotation augmentation for point cloud
        self.rotation_generator = RandomSOd(2)

        # CIFAR-10 point cloud: 3 scalar features (RGB) per patch + 2D position vectors
        patch_size = config.dataset.patch_size
        in_channels_scalar = patch_size * patch_size * 3
        in_channels_vector = 2

        # Number of patches in the point cloud
        self.avg_num_nodes = (config.dataset.image_size // patch_size) ** 2

        # Validate solid name
        solid_name = config.model.solid_name
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Invalid solid_name '{solid_name}'. "
                f"Must be one of: {list(PLATONIC_GROUPS.keys())}"
            )

        # Initialize Platonic Transformer
        self.net = PlatonicTransformer(
            # Basic specification
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.dataset.num_classes,
            output_dim_vec=0,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            # Pooling and readout
            scalar_task_level=config.model.scalar_task_level,
            vector_task_level=config.model.vector_task_level,
            ffn_readout=config.model.ffn_readout,
            # Attention block specification
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            # Positional encoding (RoPE and APE)
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
        )

        # Setup metrics
        num_classes = config.dataset.num_classes
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with optional rotation augmentation."""
        # Apply rotation augmentation during training if enabled
        if self.training and self.config.training.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
        else:
            rot = torch.eye(2, device=data.pos.device)

        # Create vector features from rotation matrix
        vec = rot.transpose(-2, -1).unsqueeze(0).expand(data.pos.shape[0], -1, -1)

        # Forward pass through the network
        pred, _ = self.net(
            data.x,
            data.pos,
            data.batch,
            vec=vec,
            avg_num_nodes=self.avg_num_nodes
        )
        return pred

    def _calculate_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on configured loss function."""
        if self.config.training.loss_fn == "bce":
            y_one_hot = torch.nn.functional.one_hot(
                y, num_classes=self.config.dataset.num_classes
            ).float()
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, y_one_hot)
        else:
            return torch.nn.functional.cross_entropy(pred, y)

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.train_metric(pred, data.y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=self.config.training.batch_size
        )
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> None:
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.valid_metric(pred, data.y)
        self.log("valid_loss", loss, batch_size=self.config.training.batch_size)

    def test_step(self, data: Data, batch_idx: int) -> None:
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("valid_acc", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log("test_acc", self.test_metric, prog_bar=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler."""

        optimizer_name = self.config.optimizer.name.lower()
        if optimizer_name == "lamb":
            if Lamb is None:
                raise ImportError(
                    "timm not installed. Cannot use LAMB optimizer. "
                    "Run: pip install timm"
                )
            optimizer = Lamb(
                self.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. "
                "Supported: 'adamw', 'lamb'"
            )
        
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.config.scheduler.warmup_epochs,
            max_iters=self.trainer.max_epochs
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_data(
    config: ml_collections.ConfigDict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 datasets with patch-based preprocessing."""
    
    if config.dataset.image_size % config.dataset.patch_size != 0:
        raise ValueError(
            f"Image size ({config.dataset.image_size}) must be divisible "
            f"by patch_size ({config.dataset.patch_size})"
        )

    if config.training.use_deit3_augmentation:
        # DeiT-III augmentation: 3-Augment + Color Jitter
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomChoice([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.RandomSolarize(threshold=128.0),
                torchvision.transforms.GaussianBlur(kernel_size=(3, 3))
            ]),
            torchvision.transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
    else:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])
    
    # Create patch position grid
    patch_size = config.dataset.patch_size
    num_patches_1d = config.dataset.image_size // patch_size
    grid = torch.linspace(0.0, 1.0, num_patches_1d)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='xy')
    patch_pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    # Zero-center positions (relevant for global RoPE)
    patch_pos = patch_pos - 0.5

    def collate_fn(batch):
        """Convert a batch of images into graph-structured point clouds."""
        data_list = []
        p = patch_size
        for image_tensor, label in batch:
            # Extract patches using unfold
            patches = image_tensor.unfold(1, p, p).unfold(2, p, p)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous()
            x = patches.view(-1, 3 * p * p)
            
            # Create Data object
            data = Data(
                x=x,
                pos=patch_pos.clone(),
                y=torch.tensor([label])
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=config.dataset.data_dir,
        train=True,
        transform=transform_train,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.dataset.data_dir,
        train=False,
        transform=transform_test,
        download=True
    )
    
    # Split training into train/val
    train_size = int(config.dataset.train_val_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
   
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main(config: ml_collections.ConfigDict) -> None:
    """Train and evaluate the Platonic Transformer on CIFAR-10."""
    
    print_config(config, "CIFAR-10 Training Configuration")
    
    pl.seed_everything(config.seed)
    
    train_loader, val_loader, test_loader = load_data(config)

    # Configure accelerator
    if config.system.gpus > 0:
        accelerator, devices = "gpu", config.system.gpus
    else:
        accelerator, devices = "cpu", "auto"
        

    if config.logging.enabled:
        save_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "logs"
        )
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            config=config.to_dict(),
            save_dir=save_dir
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last
        ),
        TimerCallback()
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))


    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        strategy=DDPStrategy(find_unused_parameters=True) if config.system.gpus > 1 else 'auto'
    )

    test_ckpt = config.testing.test_ckpt
    if test_ckpt is None:
        model = CIFAR10Model(config)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path='best')
    else:
        model = CIFAR10Model.load_from_checkpoint(test_ckpt)
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/cifar10_deit.yaml")
    args, unknown_args = parser.parse_known_args()
    
    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args  # Automatically infers parameter locations
    )

    main(config)
