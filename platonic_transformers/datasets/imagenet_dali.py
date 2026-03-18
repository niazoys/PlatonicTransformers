"""DALI ImageNet dataloader producing torch_geometric Batch objects.

GPU-fused pipeline: JPEG decode, random-resized-crop, horizontal flip,
ThreeAugment, ColorJitter, RandAugment, normalization all run inside DALI.
Post-DALI: RandomErasing, Mixup/CutMix on image tensors, then unfold into
patch-based point clouds for PlatonicTransformer.

Requires: ``pip install nvidia-dali-cuda120``
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import ml_collections
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch_geometric.data import Batch

from platonic_transformers.datasets.dali_rand_augment import dali_rand_augment


# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN_STD_BY_SIZE = {
    32: ([0.48450482, 0.45589244, 0.40366766], [0.25668961, 0.24765739, 0.26173702]),
    64: ([0.48453078, 0.45592377, 0.40370297], [0.26425716, 0.25516447, 0.26875198]),
}
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# DALI pipelines
# ---------------------------------------------------------------------------


def _solarize(images):
    """Solarize: invert pixels whose value >= 128."""
    mask = fn.cast(images >= 128, dtype=types.UINT8)
    inverted = fn.cast(255, dtype=types.UINT8) - images
    result = mask * inverted + (fn.cast(1, dtype=types.UINT8) - mask) * images
    return fn.cast(result, dtype=types.UINT8)


@pipeline_def(enable_conditionals=True)
def _train_pipeline_fused(
    file_root: str,
    image_size: int,
    final_image_size: int,
    norm_mean: tuple,
    norm_std: tuple,
    use_three_augment: bool = False,
    color_jitter: float = 0.0,
    rand_augment_config: str = "",
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Training pipeline with decode, crop, augmentations, and normalization."""
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        name="reader",
        shard_id=shard_id,
        num_shards=num_shards,
    )

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=(image_size, image_size),
        random_area=(0.08, 1.0),
        interp_type=types.INTERP_CUBIC,
    )
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    if final_image_size != image_size:
        images = fn.resize(
            images,
            size=(final_image_size, final_image_size),
            interp_type=types.INTERP_CUBIC,
        )

    # ThreeAugment (grayscale / solarize / blur, each with p=1/3)
    if use_three_augment:
        coin = fn.random.uniform(range=(0.0, 1.0))
        if coin < (1.0 / 3.0):
            grey = fn.color_space_conversion(
                images, image_type=types.RGB, output_type=types.GRAY,
            )
            images = fn.cat(grey, grey, grey, axis=2)
        else:
            if coin < (2.0 / 3.0):
                images = _solarize(images)
            else:
                sigma = fn.random.uniform(range=(0.1, 2.0))
                images = fn.gaussian_blur(images, sigma=sigma, window_size=5)

    # ColorJitter (brightness, contrast, saturation)
    if color_jitter > 0:
        brightness = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        contrast = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        saturation = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        images = fn.color_twist(
            images, brightness=brightness, contrast=contrast, saturation=saturation,
        )

    # RandAugment (timm-compatible, applied on uint8 before normalize)
    if rand_augment_config:
        images = dali_rand_augment(
            images,
            config_str=rand_augment_config,
            shape=(final_image_size, final_image_size),
        )

    # uint8 -> float32 + normalize
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in norm_mean],
        std=[s * 255.0 for s in norm_std],
    )

    return images, labels


@pipeline_def
def _val_pipeline_fused(
    file_root: str,
    image_size: int,
    final_image_size: int,
    eval_crop_ratio: float,
    norm_mean: tuple,
    norm_std: tuple,
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Validation pipeline with decode, resize, center crop, and normalization."""
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=False,
        name="reader",
        shard_id=shard_id,
        num_shards=num_shards,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    eval_size = int(image_size / eval_crop_ratio)
    images = fn.resize(images, resize_shorter=eval_size, interp_type=types.INTERP_CUBIC)
    images = fn.crop(images, crop=(image_size, image_size))

    if final_image_size != image_size:
        images = fn.resize(
            images,
            size=(final_image_size, final_image_size),
            interp_type=types.INTERP_CUBIC,
        )

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in norm_mean],
        std=[s * 255.0 for s in norm_std],
    )

    return images, labels


# ---------------------------------------------------------------------------
# DALI -> torch_geometric Batch wrapper
# ---------------------------------------------------------------------------


class _DALIGraphBatchWrapper:
    """Wraps DALIGenericIterator to yield torch_geometric Batch objects.

    Converts DALI's (B, C, H, W) GPU image tensors into patch-based point
    clouds with:
      - x: (B*num_patches, 3*p*p) flattened patch features
      - pos: (B*num_patches, 2) zero-centered grid positions
      - y: (B,) or (B, num_classes) labels
      - batch: (B*num_patches,) batch membership indices
    """

    def __init__(
        self,
        dali_iterator: DALIGenericIterator,
        patch_size: int,
        image_size: int,
        mixup_fn=None,
        random_erasing_fn=None,
        training: bool = True,
    ):
        self._iter = dali_iterator
        self.patch_size = patch_size
        self.training = training
        self.mixup_fn = mixup_fn
        self.random_erasing_fn = random_erasing_fn

        # Pre-compute fixed position grid (created once, reused every batch)
        num_patches_1d = image_size // patch_size
        grid = torch.linspace(0.0, 1.0, num_patches_1d)
        grid_x, grid_y = torch.meshgrid(grid, grid, indexing="xy")
        self._patch_pos_cpu = (
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1) - 0.5
        )
        self._num_patches = num_patches_1d * num_patches_1d
        self._patch_pos_gpu = None

    def _images_to_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Batch:
        """Convert (B, C, H, W) tensor to torch_geometric Batch."""
        B = images.shape[0]
        p = self.patch_size

        # Unfold into patches: (B, 3, H, W) -> (B, H//p, W//p, 3, p, p)
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = patches.view(B * self._num_patches, 3 * p * p)

        # Expand pre-computed position grid
        if self._patch_pos_gpu is None or self._patch_pos_gpu.device != images.device:
            self._patch_pos_gpu = self._patch_pos_cpu.to(images.device)
        pos = self._patch_pos_gpu.unsqueeze(0).expand(B, -1, -1).reshape(
            B * self._num_patches, 2
        )

        # Batch indices
        batch_idx = (
            torch.arange(B, device=images.device)
            .unsqueeze(1)
            .expand(-1, self._num_patches)
            .reshape(-1)
        )

        return Batch(
            x=x,
            pos=pos,
            y=labels,
            batch=batch_idx,
            num_nodes=B * self._num_patches,
        )

    def __iter__(self):
        for batch in self._iter:
            data = batch[0]
            images = data["images"]  # (B, C, H, W) float32 GPU
            labels = data["labels"].squeeze(-1).long()

            # Apply augmentations that need image-level spatial structure
            if self.training:
                if self.random_erasing_fn is not None:
                    images = self.random_erasing_fn(images)
                if self.mixup_fn is not None:
                    images, labels = self.mixup_fn(images, labels)

            yield self._images_to_batch(images, labels)

    def __len__(self):
        return len(self._iter)


# ---------------------------------------------------------------------------
# NVMe staging
# ---------------------------------------------------------------------------


def _stage_to_local(src: Path, dst: Path) -> Path:
    """Copy ImageFolder data to fast local storage (e.g. NVMe).

    Idempotent: skips copy if train/ and val/ already exist at dst.
    """
    print(f"[data-staging] local_staging_dir={dst}, checking ...", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(dst).free
    min_bytes = 160 * (1024**3)
    if free_bytes < min_bytes:
        raise RuntimeError(
            f"[data-staging] {dst} has only "
            f"{free_bytes / (1024**3):.1f} GB free (need {min_bytes / (1024**3):.0f} GB)"
        )

    if (dst / "train").is_dir() and (dst / "val").is_dir():
        print(
            f"[data-staging] {dst} already staged (train/ and val/ found), skipping copy.",
            flush=True,
        )
        return dst

    print(
        f"[data-staging] Copying {src} -> {dst} (this may take 10-20 min) ...",
        flush=True,
    )
    result = subprocess.run(
        ["cp", "-a", "--no-clobber", "-r", str(src / "train"), str(src / "val"), str(dst)],
        check=False,
        timeout=3600,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"[data-staging] cp failed with exit code {result.returncode}"
        )
    print(f"[data-staging] Done. Using local path: {dst}", flush=True)
    return dst


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(
    config: ml_collections.ConfigDict,
) -> Tuple["_DALIGraphBatchWrapper", "_DALIGraphBatchWrapper", "_DALIGraphBatchWrapper"]:
    """Build DALI-backed ImageNet train/val/test loaders returning graph Batches.

    Args:
        config: experiment config with dataset, dali, augmentation, mixup sections.

    Returns:
        (train_loader, val_loader, test_loader) where each yields
        torch_geometric Batch objects on GPU.
    """
    data_dir = Path(config.dataset.data_dir)
    image_size = config.dataset.image_size
    final_image_size = getattr(config.dataset, "final_image_size", image_size)
    patch_size = config.dataset.patch_size
    eval_crop_ratio = getattr(config.dataset, "eval_crop_ratio", 0.875)
    batch_size = config.training.batch_size

    # DALI settings
    dali_cfg = config.dali
    num_workers = dali_cfg.num_workers
    prefetch_factor = getattr(dali_cfg, "prefetch_factor", 2)
    seed = getattr(dali_cfg, "seed", config.seed)

    # Optional NVMe staging
    local_staging_dir = getattr(config.dataset, "local_staging_dir", None)
    if local_staging_dir:
        data_dir = _stage_to_local(data_dir, Path(local_staging_dir))

    train_root = str(data_dir / "train")
    val_root = str(data_dir / "val")

    # Normalization
    mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
        final_image_size, (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD)
    )
    norm_mean = tuple(mean)
    norm_std = tuple(std)

    # Augmentation config
    aug_cfg = config.augmentation
    use_three_augment = getattr(aug_cfg, "use_three_augment", False)
    color_jitter = getattr(aug_cfg, "color_jitter", 0.0)
    rand_augment = getattr(aug_cfg, "rand_augment", "") or ""
    random_erasing_prob = getattr(aug_cfg, "random_erasing_prob", 0.0)
    random_erasing_mode = getattr(aug_cfg, "random_erasing_mode", "pixel")

    # Distributed sharding
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Build DALI pipelines
    train_pipe = _train_pipeline_fused(
        file_root=train_root,
        image_size=image_size,
        final_image_size=final_image_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        use_three_augment=use_three_augment,
        color_jitter=color_jitter,
        rand_augment_config=rand_augment,
        shard_id=local_rank,
        num_shards=world_size,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=local_rank,
        seed=seed,
        prefetch_queue_depth=prefetch_factor,
    )
    train_pipe.build()

    val_pipe = _val_pipeline_fused(
        file_root=val_root,
        image_size=image_size,
        final_image_size=final_image_size,
        eval_crop_ratio=eval_crop_ratio,
        norm_mean=norm_mean,
        norm_std=norm_std,
        shard_id=local_rank,
        num_shards=world_size,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=local_rank,
        seed=seed,
        prefetch_queue_depth=prefetch_factor,
    )
    val_pipe.build()

    # RandomErasing (applied on GPU before patchification)
    random_erasing_fn = None
    if random_erasing_prob > 0:
        from timm.data.random_erasing import RandomErasing

        random_erasing_fn = RandomErasing(
            probability=random_erasing_prob,
            mode=random_erasing_mode,
            device="cuda",
        )

    # Mixup/CutMix (applied on GPU before patchification)
    mixup_fn = None
    mixup_cfg = getattr(config, "mixup", None)
    if mixup_cfg is not None:
        mixup_alpha = getattr(mixup_cfg, "mixup_alpha", 0.0)
        cutmix_alpha = getattr(mixup_cfg, "cutmix_alpha", 0.0)
        if mixup_alpha > 0 or cutmix_alpha > 0:
            from timm.data import Mixup

            mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=getattr(mixup_cfg, "prob", 1.0),
                switch_prob=getattr(mixup_cfg, "switch_prob", 0.5),
                mode=getattr(mixup_cfg, "mode", "batch"),
                label_smoothing=getattr(mixup_cfg, "label_smoothing", 0.1),
                num_classes=config.dataset.num_classes,
            )

    # Wrap DALI iterators
    train_loader = _DALIGraphBatchWrapper(
        DALIGenericIterator(
            train_pipe,
            output_map=["images", "labels"],
            reader_name="reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        ),
        patch_size=patch_size,
        image_size=final_image_size,
        mixup_fn=mixup_fn,
        random_erasing_fn=random_erasing_fn,
        training=True,
    )

    val_iter = DALIGenericIterator(
        val_pipe,
        output_map=["images", "labels"],
        reader_name="reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )
    val_loader = _DALIGraphBatchWrapper(
        val_iter,
        patch_size=patch_size,
        image_size=final_image_size,
        training=False,
    )

    # Test uses the same pipeline as validation
    test_loader = val_loader

    return train_loader, val_loader, test_loader
