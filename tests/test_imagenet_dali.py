"""Test script for the ImageNet DALI dataloader.

Builds the DALI pipeline, loads a few batches, and validates shapes,
dtypes, and basic semantics of the torch_geometric Batch objects.

Usage:
    python tests/test_imagenet_dali.py --data_dir /path/to/imagenet [--batch_size 32]

Requires a GPU and an ImageNet directory with train/ and val/ subdirectories.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import torch


def build_config(data_dir: str, batch_size: int = 32, image_size: int = 224,
                 patch_size: int = 16) -> ml_collections.ConfigDict:
    """Create a minimal config for testing the DALI loader."""
    config = ml_collections.ConfigDict()
    config.seed = 42

    config.dataset = ml_collections.ConfigDict()
    config.dataset.data_dir = data_dir
    config.dataset.num_classes = 1000
    config.dataset.image_size = image_size
    config.dataset.final_image_size = image_size
    config.dataset.patch_size = patch_size
    config.dataset.eval_crop_ratio = 0.875
    config.dataset.local_staging_dir = None

    config.dali = ml_collections.ConfigDict()
    config.dali.num_workers = 4
    config.dali.prefetch_factor = 2
    config.dali.seed = 42

    config.augmentation = ml_collections.ConfigDict()
    config.augmentation.use_three_augment = False
    config.augmentation.color_jitter = 0.0
    config.augmentation.rand_augment = ""
    config.augmentation.random_erasing_prob = 0.0
    config.augmentation.random_erasing_mode = "pixel"

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = batch_size

    return config


def check_batch(batch, batch_size: int, num_patches: int, patch_dim: int,
                split: str, batch_idx: int):
    """Validate a single torch_geometric Batch object."""
    B = batch.y.shape[0]
    expected_nodes = B * num_patches

    print(f"  [{split}] batch {batch_idx}: B={B}, "
          f"x={tuple(batch.x.shape)}, pos={tuple(batch.pos.shape)}, "
          f"y={tuple(batch.y.shape)}, batch_idx={tuple(batch.batch.shape)}")

    # x: flattened patch features
    assert batch.x.shape == (expected_nodes, patch_dim), \
        f"x shape mismatch: {batch.x.shape} != ({expected_nodes}, {patch_dim})"
    assert batch.x.dtype == torch.float32, f"x dtype: {batch.x.dtype}"

    # pos: 2D patch positions
    assert batch.pos.shape == (expected_nodes, 2), \
        f"pos shape mismatch: {batch.pos.shape} != ({expected_nodes}, 2)"
    assert batch.pos.dtype == torch.float32, f"pos dtype: {batch.pos.dtype}"

    # y: class labels
    assert batch.y.ndim == 1, f"y should be 1D, got {batch.y.ndim}D"
    assert batch.y.dtype == torch.int64, f"y dtype: {batch.y.dtype}"
    assert (batch.y >= 0).all() and (batch.y < 1000).all(), \
        f"y out of range: min={batch.y.min()}, max={batch.y.max()}"

    # batch indices
    assert batch.batch.shape == (expected_nodes,), \
        f"batch shape mismatch: {batch.batch.shape}"
    assert batch.batch.min() == 0, f"batch min: {batch.batch.min()}"
    assert batch.batch.max() == B - 1, f"batch max: {batch.batch.max()}"

    # pos should be zero-centered (range roughly [-0.5, 0.5])
    assert batch.pos.min() >= -0.6, f"pos min too low: {batch.pos.min()}"
    assert batch.pos.max() <= 0.6, f"pos max too high: {batch.pos.max()}"

    # x should be normalized (not raw uint8 range)
    assert batch.x.abs().max() < 100, \
        f"x values look unnormalized: max abs = {batch.x.abs().max()}"

    # Data should be on GPU
    assert batch.x.is_cuda, "x should be on GPU"
    assert batch.pos.is_cuda, "pos should be on GPU"

    print(f"          x range: [{batch.x.min():.3f}, {batch.x.max():.3f}], "
          f"pos range: [{batch.pos.min():.3f}, {batch.pos.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Test ImageNet DALI dataloader")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet directory (with train/ and val/)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=3,
                        help="Number of batches to test per split")
    args = parser.parse_args()

    num_patches_1d = args.image_size // args.patch_size
    num_patches = num_patches_1d ** 2
    patch_dim = 3 * args.patch_size * args.patch_size

    print(f"Config: image_size={args.image_size}, patch_size={args.patch_size}, "
          f"num_patches={num_patches} ({num_patches_1d}x{num_patches_1d}), "
          f"patch_dim={patch_dim}, batch_size={args.batch_size}")
    print(f"Data dir: {args.data_dir}")
    print()

    # Verify directories exist
    for split in ("train", "val"):
        split_dir = os.path.join(args.data_dir, split)
        assert os.path.isdir(split_dir), f"Missing directory: {split_dir}"
    print("Directory structure OK")

    # Build config and load data
    config = build_config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )

    print("Building DALI pipelines ...")
    t0 = time.time()
    from platonic_transformers.datasets.imagenet_dali import load_data
    train_loader, val_loader, test_loader = load_data(config)
    print(f"Pipelines built in {time.time() - t0:.1f}s")
    print(f"Train loader length: {len(train_loader)} batches")
    print(f"Val loader length:   {len(val_loader)} batches")
    print()

    # --- Test training split ---
    print(f"=== Training split (first {args.num_batches} batches) ===")
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        check_batch(batch, args.batch_size, num_patches, patch_dim, "train", i)
        if i + 1 >= args.num_batches:
            break
    print(f"Train iteration: {time.time() - t0:.2f}s for {args.num_batches} batches")
    print()

    # --- Test validation split ---
    print(f"=== Validation split (first {args.num_batches} batches) ===")
    t0 = time.time()
    for i, batch in enumerate(val_loader):
        check_batch(batch, args.batch_size, num_patches, patch_dim, "val", i)
        if i + 1 >= args.num_batches:
            break
    print(f"Val iteration: {time.time() - t0:.2f}s for {args.num_batches} batches")
    print()

    # --- Test that test_loader is the same as val_loader ---
    assert test_loader is val_loader, "test_loader should be val_loader"
    print("test_loader is val_loader: OK")
    print()

    # --- Throughput benchmark (train) ---
    print("=== Throughput benchmark (train, 10 batches) ===")
    num_bench = 10
    torch.cuda.synchronize()
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        if i + 1 >= num_bench:
            break
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    imgs_per_sec = num_bench * args.batch_size / elapsed
    print(f"{num_bench} batches in {elapsed:.2f}s => {imgs_per_sec:.0f} images/s")
    print()

    print("All tests passed!")


if __name__ == "__main__":
    main()
