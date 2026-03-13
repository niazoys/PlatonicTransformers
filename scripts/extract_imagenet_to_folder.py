"""Extract ImageNet from HuggingFace Arrow format to torchvision ImageFolder format.

Writes raw JPEG bytes directly (no re-encoding) so images are bit-identical
to what HF datasets stores.  Uses zero-padded integer class IDs as folder
names so that ImageFolder's sorted-order class assignment matches the HF
label indices exactly.

Output layout:
    {output_dir}/train/0000/000000.jpg   (label=0, sample 0)
    {output_dir}/train/0000/000001.jpg   (label=0, sample 1)
    {output_dir}/train/0999/004231.jpg   (label=999, sample 4231)
    {output_dir}/val/0000/...

After extraction, runs a verification pass comparing random samples from
both the HF dataset and the new ImageFolder to confirm pixel-exact match
and correct label alignment.

Usage:
    PYTHONPATH=. python scripts/extract_imagenet_to_folder.py
"""

import io
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import datasets
import numpy as np
from PIL import Image

HF_DATASET = "ILSVRC/imagenet-1k"
HF_CACHE = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
OUTPUT_DIR = "/shared/data/image_datasets/imagenet_folder"
NUM_VERIFY_SAMPLES = 200


def extract_split(split: str) -> None:
    """Extract one split (train or validation) to ImageFolder layout."""
    print(f"\n{'='*60}")
    print(f"Extracting split: {split}")
    print(f"{'='*60}")

    hf_token = os.environ.get("HF_TOKEN")

    ds = datasets.load_dataset(
        HF_DATASET, split=split, streaming=False,
        cache_dir=HF_CACHE, token=hf_token,
    )
    ds_raw = ds.cast_column("image", datasets.Image(decode=False))

    folder_name = "train" if split == "train" else "val"
    out_root = Path(OUTPUT_DIR) / folder_name

    class_counters = defaultdict(int)
    total = len(ds_raw)
    t0 = time.time()

    for i in range(total):
        row = ds_raw[i]
        label = row["label"]
        raw_image = row["image"]

        class_dir = out_root / f"{label:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        img_bytes = raw_image["bytes"]
        ext = _guess_ext(img_bytes)
        filename = f"{class_counters[label]:06d}{ext}"
        filepath = class_dir / filename

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        class_counters[label] += 1

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{split}] {i+1:>8d}/{total}  "
                  f"({rate:.0f} img/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - t0
    num_classes = len(class_counters)
    print(f"  [{split}] Done: {total} images, {num_classes} classes "
          f"in {elapsed:.0f}s ({total/elapsed:.0f} img/s)")

    return total, num_classes


def _guess_ext(raw_bytes: bytes) -> str:
    """Determine file extension from magic bytes."""
    if raw_bytes[:2] == b'\xff\xd8':
        return ".jpg"
    if raw_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return ".png"
    if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        return ".webp"
    return ".jpg"


def verify_split(split: str) -> bool:
    """Verify that ImageFolder labels and pixels match the HF dataset."""
    print(f"\n{'='*60}")
    print(f"Verifying split: {split}")
    print(f"{'='*60}")

    hf_token = os.environ.get("HF_TOKEN")

    ds_hf = datasets.load_dataset(
        HF_DATASET, split=split, streaming=False,
        cache_dir=HF_CACHE, token=hf_token,
    )

    folder_name = "train" if split == "train" else "val"
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    ds_folder = ImageFolder(
        str(Path(OUTPUT_DIR) / folder_name),
        transform=transforms.ToTensor(),
    )

    # Verify dataset sizes match
    if len(ds_hf) != len(ds_folder):
        print(f"  SIZE MISMATCH: HF={len(ds_hf)}, Folder={len(ds_folder)}")
        return False
    print(f"  Size match: {len(ds_hf)} images")

    # Verify class count
    num_hf_classes = len(set(ds_hf["label"]))
    num_folder_classes = len(ds_folder.classes)
    if num_hf_classes != num_folder_classes:
        print(f"  CLASS COUNT MISMATCH: HF={num_hf_classes}, Folder={num_folder_classes}")
        return False
    print(f"  Class count match: {num_hf_classes}")

    # Verify label mapping: folder "0042" should map to class_idx 42
    for class_idx, class_name in enumerate(ds_folder.classes):
        expected = f"{class_idx:04d}"
        if class_name != expected:
            print(f"  LABEL MAPPING ERROR: class_idx={class_idx}, "
                  f"folder={class_name}, expected={expected}")
            return False
    print(f"  Label mapping verified: folder names match class indices")

    # Spot-check random samples: compare HF pixels to saved file pixels
    rng = np.random.RandomState(42)

    # Build HF class-to-indices mapping
    hf_class_indices = defaultdict(list)
    for i, label in enumerate(ds_hf["label"]):
        hf_class_indices[label].append(i)

    # Build Folder class-to-indices mapping
    folder_class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(ds_folder.samples):
        folder_class_indices[label].append(idx)

    n_checked = 0
    n_pixel_match = 0
    n_label_match = 0
    to_tensor = transforms.ToTensor()

    sample_indices = rng.choice(len(ds_hf), size=min(NUM_VERIFY_SAMPLES, len(ds_hf)), replace=False)

    for hf_idx in sample_indices:
        hf_row = ds_hf[int(hf_idx)]
        hf_label = hf_row["label"]
        hf_img = hf_row["image"].convert("RGB")

        # Find which intra-class position this sample is
        intra_idx = hf_class_indices[hf_label].index(int(hf_idx))

        # Get the corresponding ImageFolder sample
        folder_idx = folder_class_indices[hf_label][intra_idx]
        folder_img, folder_label = ds_folder[folder_idx]

        # Check label
        if hf_label == folder_label:
            n_label_match += 1

        # Check pixels (compare as tensors)
        hf_tensor = to_tensor(hf_img)
        if hf_tensor.shape == folder_img.shape and (hf_tensor == folder_img).all():
            n_pixel_match += 1

        n_checked += 1

    print(f"  Spot-checked {n_checked} random samples:")
    print(f"    Label match:  {n_label_match}/{n_checked}")
    print(f"    Pixel match:  {n_pixel_match}/{n_checked}")

    ok = (n_label_match == n_checked) and (n_pixel_match == n_checked)
    if ok:
        print(f"  VERIFICATION PASSED")
    else:
        print(f"  VERIFICATION FAILED")
    return ok


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "validation"]:
        extract_split(split)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE — starting verification")
    print("=" * 60)

    all_ok = True
    for split in ["train", "validation"]:
        if not verify_split(split):
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL VERIFICATIONS PASSED")
        print(f"ImageFolder dataset ready at: {OUTPUT_DIR}")
    else:
        print("VERIFICATION FAILED — check output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
