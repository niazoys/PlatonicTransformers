import os
import sys
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import pytorch_lightning as pl
import torch
import torch_geometric.nn as tgnn
import torch_geometric.transforms as T
import torchmetrics
from pytorch_lightning.callbacks import Timer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomScale

from platonic_transformers.datasets.scanobjectnn import ScanObjectNN
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.utils.utils import (
    CosineWarmupScheduler,
    RandomJitter,
    RandomSO2AroundAxis,
    RandomSOd,
)
from mains._optim import make_param_groups

# Performance optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy("file_system")


def patchify(data: Data, sampling_ratio: float, num_neighbours: int):
    """FPS centers + KNN local patches → vector features per center.

    Returns center_ids too so callers can subsample any per-point fields
    (e.g. data.heights) that would otherwise stay at the original N.
    """
    center_ids = tgnn.fps(data.pos, data.batch, sampling_ratio)
    centers = data.pos[center_ids]
    batch_centers = data.batch[center_ids]
    knn_ids = tgnn.knn(
        x=data.pos,
        y=centers,
        batch_x=data.batch,
        batch_y=batch_centers,
        k=num_neighbours,
    )
    vectors = data.pos[knn_ids[0]].reshape(-1, num_neighbours, 3) - centers[:, None]
    return centers, vectors, batch_centers, center_ids


class ScanObjectNNModel(pl.LightningModule):
    """Lightning module for ScanObjectNN PB_T50_RS classification."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        if config.augmentation.inplane_augm:
            self.rotation_generator = RandomSO2AroundAxis(axis=1, degrees=180)
        else:
            self.rotation_generator = RandomSOd(3)

        self.avg_num_nodes = (
            config.model.patchify_num_centers
            if config.model.patchify
            else config.dataset.num_points
        )

        # Vector input channels: 1 channel for the rotated up-vector (rot @ e_up).
        # Dataloader-side FPS+KNN patchify adds K neighbour offsets per center;
        # in-model EdgeConv patchify replaces that step (no fixed K channels).
        in_channels_scalar = 0
        in_channels_vector = 1  # up-vector
        edge_conv = config.model.get("edge_conv_patchify", False)
        if config.model.patchify and not edge_conv:
            in_channels_vector += config.model.patchify_num_neighbours

        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Unsupported solid_name '{solid_name}'. "
                f"Available: {list(PLATONIC_GROUPS.keys())}"
            )

        num_classes = 15
        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=config.model.hidden_dim,
            output_dim=num_classes,
            output_dim_vec=0,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            scalar_task_level=config.model.scalar_task_level,
            vector_task_level=config.model.vector_task_level,
            ffn_readout=config.model.ffn_readout,
            trivial_readout=config.model.get("trivial_readout", False),
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
            edge_conv_patchify=config.model.get("edge_conv_patchify", False),
            edge_conv_k=config.model.get("edge_conv_k", 32),
            # FPS sampling ratio for in-model EdgeConv patchify, derived from
            # patchify_num_centers / num_points so it stays consistent with the
            # standard dataloader-side patchify path.
            fps_ratio=config.model.patchify_num_centers / config.dataset.num_points,
        )

        if config.model.get("compile", True):
            self.net = torch.compile(self.net)

        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        # Macro-averaged (per-class mean) accuracy = mAcc as reported by Point Cloud
        # Mamba and PointNeXt. PB_T50_RS is class-imbalanced so OA > mAcc usually.
        self.valid_metric_macc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_metric_macc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, data: Data) -> torch.Tensor:
        train_augm = self.training and self.config.augmentation.train_augm
        test_augm = (not self.training) and self.config.augmentation.test_augm
        scale_in_forward = self.config.augmentation.get("scale_in_forward", False)
        per_sample_augm = self.config.augmentation.get("per_sample_augm", False)
        s_lo = self.config.augmentation.min_scale
        s_hi = self.config.augmentation.max_scale

        # Build the per-node combined transform T_node = s_node * rot_node.
        #   per_sample_augm=true  → independent (rot, scale) per object in the batch
        #   per_sample_augm=false → one (rot, scale) shared across the whole batch (legacy)
        if per_sample_augm and train_augm:
            n_obj = int(data.batch.max().item()) + 1
            rot_per_obj = self.rotation_generator(n=n_obj).type_as(data.pos)             # [B, 3, 3]
            if scale_in_forward:
                s_per_obj = torch.empty(n_obj, device=data.pos.device, dtype=data.pos.dtype).uniform_(s_lo, s_hi)
            else:
                s_per_obj = torch.ones(n_obj, device=data.pos.device, dtype=data.pos.dtype)
            # Apply per-node: pos_new = s * R @ pos
            rot_per_node = rot_per_obj[data.batch]                                       # [N, 3, 3]
            s_per_node   = s_per_obj[data.batch]                                         # [N]
            data.pos = s_per_node[:, None] * torch.einsum('nij,nj->ni', rot_per_node, data.pos)
            if hasattr(data, 'normal') and data.normal is not None:
                data.normal = torch.einsum('nij,nj->ni', rot_per_node, data.normal)
            rot_scaled_per_obj = s_per_obj[:, None, None] * rot_per_obj                  # [B, 3, 3]
        else:
            # Per-batch (legacy) path
            if scale_in_forward and train_augm:
                s = torch.empty((), device=data.pos.device, dtype=data.pos.dtype).uniform_(s_lo, s_hi)
                data.pos = data.pos * s
            else:
                s = torch.tensor(1.0, device=data.pos.device, dtype=data.pos.dtype)
            if train_augm or test_augm:
                rot = self.rotation_generator().type_as(data.pos)
                data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
                if hasattr(data, 'normal') and data.normal is not None:
                    data.normal = torch.einsum('ij,bj->bi', rot, data.normal)
            else:
                rot = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
            rot_scaled = s * rot                                                          # [3, 3]

        vec_parts = []

        # In-model EdgeConv patchify operates on the dense cloud, so skip the
        # dataloader-side FPS+KNN+stacked-offsets when it's enabled.
        _internal_patch = self.config.model.get("edge_conv_patchify", False)
        if self.config.model.patchify and not _internal_patch:
            centers, vectors, batch_centers, center_ids = patchify(
                data,
                sampling_ratio=self.config.model.patchify_num_centers / self.config.dataset.num_points,
                num_neighbours=self.config.model.patchify_num_neighbours,
            )
            data.pos = centers
            data.batch = batch_centers
            vec_parts.append(vectors)

        # Up-vector vector input: 1 channel = rot @ e_up. The model needs *some*
        # signal for which way is up after the random rotation; passing the
        # rotated up-vector is the minimal symmetry-breaking input.
        # ScanObjectNN is Y-up (axis 1).
        up_axis = self.config.augmentation.get("up_axis", 1)
        e_up = torch.zeros(3, device=data.pos.device, dtype=data.pos.dtype); e_up[up_axis] = 1.0
        if per_sample_augm and train_augm:
            up_per_obj = torch.einsum('bij,j->bi', rot_scaled_per_obj, e_up)              # [B, 3]
            up_per_node = up_per_obj[data.batch]                                          # [N, 3]
            vec_parts.append(up_per_node[:, None, :])                                     # [N, 1, 3]
        else:
            up_vec = rot_scaled @ e_up                                                    # [3]
            vec_parts.append(up_vec.view(1, 1, 3).expand(data.pos.shape[0], -1, -1))      # [N, 1, 3]

        x = None
        vec = torch.cat(vec_parts, dim=1)

        pred, _ = self.net(x, data.pos, data.batch, vec=vec, avg_num_nodes=self.avg_num_nodes)
        return pred

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data)
        loss = torch.nn.functional.cross_entropy(
            pred, data.y, label_smoothing=self.config.training.label_smoothing
        )
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.config.training.batch_size)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> None:
        pred = self(data)
        self.valid_metric(pred, data.y)
        self.valid_metric_macc(pred, data.y)

    def test_step(self, data: Data, batch_idx: int) -> None:
        # Optional test-time augmentation: average logits over N rotations.
        # config.testing.repeats=1 disables TTA (default).
        n_repeats = self.config.testing.get("repeats", 1)
        if n_repeats <= 1 or self.config.augmentation.test_augm:
            pred = self(data)
        else:
            preds = []
            original_pos = data.pos.clone()
            saved_test_augm = self.config.augmentation.test_augm
            self.config.augmentation.test_augm = True  # force rotation per pass
            for _ in range(n_repeats):
                data.pos = original_pos.clone()
                preds.append(self(data))
            self.config.augmentation.test_augm = saved_test_augm
            data.pos = original_pos
            pred = torch.stack(preds).mean(dim=0)
        self.test_metric(pred, data.y)
        self.test_metric_macc(pred, data.y)

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("valid_acc", self.valid_metric, prog_bar=True)
        self.log("valid_macc", self.valid_metric_macc)

    def on_test_epoch_end(self) -> None:
        suffix = "_rotated" if self.config.augmentation.test_augm else ""
        self.log(f"test_acc{suffix}", self.test_metric, prog_bar=True)
        self.log(f"test_macc{suffix}", self.test_metric_macc)

    def configure_optimizers(self) -> dict:
        param_groups = make_param_groups(self, self.config.optimizer.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.config.optimizer.lr)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.config.scheduler.warmup_epochs,
            max_iters=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_data(config: ml_collections.ConfigDict) -> Tuple[DataLoader, DataLoader]:
    """ScanObjectNN train/test dataloaders. The PB_T50_RS variant has no
    held-out validation split; we use the test set as both validation and
    final-test (paper convention)."""

    aug = config.augmentation
    # Scaling: anisotropic (per-axis) matches PCM; isotropic is RandomScale.
    # When scale_in_forward=true, scale aug is applied per-batch in forward
    # (so it can be baked into the reference-frame vectors). Skip dataloader scale.
    train_xforms = []
    if not aug.get("scale_in_forward", False):
        if aug.get("anisotropic_scale", False):
            train_xforms.append(AnisotropicScale(aug.min_scale, aug.max_scale))
        else:
            train_xforms.append(RandomScale((aug.min_scale, aug.max_scale)))
    if aug.noise_strength > 0:
        train_xforms.append(RandomJitter(sigma=aug.noise_strength, clip=5 * aug.noise_strength, relative=True))
    test_xforms = []
    # Center-and-normalize (PCM-style) is applied to BOTH train and test, so the
    # train and test data live on the same scale. Heights channel is emitted
    # from the *raw* gravity axis before centering.
    if aug.get("center_and_normalize", False):
        cn = CenterAndNormalize(
            gravity_dim=aug.get("up_axis", 1),
            with_heights=aug.get("use_heights", False),
        )
        train_xforms.append(cn)
        test_xforms.append(cn)
    train_transform = T.Compose(train_xforms)
    test_transform = T.Compose(test_xforms)

    # Per-subset version override lets us train on the unaugmented base
    # (clean source of augmentation) while testing on the standard
    # _augmentedrot_scale75 test set (apples-to-apples vs paper).
    train_version = config.dataset.get("data_version_train", config.dataset.data_version)
    test_version  = config.dataset.get("data_version_test",  config.dataset.data_version)
    train_dataset = ScanObjectNN(
        config.dataset.data_dir,
        subset='train',
        split=config.dataset.data_split,
        version=train_version,
        transform=train_transform,
        nbr_pts=config.dataset.num_points,
    )
    test_dataset = ScanObjectNN(
        config.dataset.data_dir,
        subset='test',
        split=config.dataset.data_split,
        version=test_version,
        transform=test_transform,
        nbr_pts=config.dataset.num_points,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "ScanObjectNN Training Configuration")
    pl.seed_everything(config.seed)

    torch.set_float32_matmul_precision(
        config.system.get("float32_matmul_precision", "medium")
    )

    train_loader, test_loader = load_data(config)

    if config.system.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.system.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        os.makedirs(save_dir, exist_ok=True)
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            entity=config.logging.get("wandb_identity", None),
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if config.system.timer is not None:
        callbacks.append(Timer(duration=config.system.timer))

    # Optional EMA on weights — ponita-style light smoothing (decay=0.99 ≈ 70-step
    # half-life). Disabled by default; flip via --training.ema_enabled=true.
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(
            decay=config.training.get("ema_decay", 0.99),
            warmup_steps=config.training.get("ema_warmup_steps", 0),
        ))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
    )

    test_ckpt = config.testing.test_ckpt
    if test_ckpt is None:
        model = ScanObjectNNModel(config)
        resume_ckpt = config.testing.resume_ckpt
        # Per paper: train on train, evaluate against test (no held-out val split).
        trainer.fit(model, train_loader, test_loader, ckpt_path=resume_ckpt)
        # Use the best-by-valid_acc checkpoint (config.checkpoint.monitor) for
        # the reported test number. Falls back to last if no monitored ckpt was
        # tracked (e.g. fast smoke runs that didn't reach a validation step).
        eval_ckpt = callbacks[0].best_model_path or callbacks[0].last_model_path or "last"
        trainer.test(model, test_loader, ckpt_path=eval_ckpt)
        # Rotated-test pass for symmetry analysis.
        model.config.augmentation.test_augm = True
        trainer.test(model, test_loader, ckpt_path=eval_ckpt)
        model.config.augmentation.test_augm = False
    else:
        model = ScanObjectNNModel.load_from_checkpoint(test_ckpt)
        model.config = config
        trainer.test(model, test_loader)
        model.config.augmentation.test_augm = True
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/scanobjectnn.yaml")
    args, unknown_args = parser.parse_known_args()

    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args,
    )

    if config.augmentation.inplane_augm and not config.augmentation.train_augm:
        raise ValueError("Cannot have inplane_augm without train_augm")

    main(config)
