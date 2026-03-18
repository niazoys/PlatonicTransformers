"""DALI RandAugment matching timm's implementation.

Provides exact replication of timm's ``rand_augment_transform`` including:
- Per-operation magnitude noise (``mstd``)
- Per-operation application probability (``p``)
- The ``_RAND_INCREASING_TRANSFORMS`` suite (``inc=1``)
- Correct magnitude scaling (timm's ``_LEVEL_DENOM=10``)

Uses DALI's native augmentation infrastructure (``auto_aug``) under the hood,
bypassing ``apply_rand_augment``'s ``isinstance(m, int)`` restriction to allow
``DataNode`` magnitude bins for per-sample magnitude noise.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

from nvidia.dali import fn, types
from nvidia.dali import math as dali_math
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core import _Augmentation, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import (
    get_translations,
    pretty_select,
)
from nvidia.dali.data_node import DataNode as _DataNode


_TIMM_LEVEL_DENOM = 10
_DEFAULT_NUM_MAGNITUDE_BINS = 31


# ---------------------------------------------------------------------------
# Config string parsing
# ---------------------------------------------------------------------------


def parse_rand_augment_config(config_str: str) -> Dict[str, Union[int, float, bool]]:
    """Parse a timm-style RandAugment config string.

    Example: ``'rand-m9-mstd0.5-inc1'`` produces
    ``{'n': 2, 'm': 9, 'mstd': 0.5, 'inc': True, 'p': 0.5}``.

    Parameters match timm defaults when omitted:
    n=2, m=10, mstd=0.0, inc=False, p=0.5.
    """
    parts = config_str.split("-")
    if parts[0] != "rand":
        raise ValueError(f"Expected 'rand' prefix, got '{parts[0]}'")

    result: Dict[str, Union[int, float, bool]] = {
        "n": 2,
        "m": 10,
        "mstd": 0.0,
        "inc": False,
        "p": 0.5,
    }

    for part in parts[1:]:
        match = re.split(r"(\d.*)", part, maxsplit=1)
        if len(match) < 2:
            continue
        key, val = match[0], match[1]
        if key == "m":
            result["m"] = int(val)
        elif key == "n":
            result["n"] = int(val)
        elif key == "mstd":
            result["mstd"] = float(val)
            if result["mstd"] > 100:
                result["mstd"] = float("inf")
        elif key == "inc":
            result["inc"] = bool(int(val))
        elif key == "p":
            result["p"] = float(val)

    return result


def _timm_m_to_dali(m: int, num_bins: int = _DEFAULT_NUM_MAGNITUDE_BINS) -> int:
    """Convert timm magnitude (0-based, scale 0.._LEVEL_DENOM) to a DALI bin index."""
    return round(m * (num_bins - 1) / _TIMM_LEVEL_DENOM)


def _timm_mstd_to_dali(mstd: float, num_bins: int = _DEFAULT_NUM_MAGNITUDE_BINS) -> float:
    """Convert timm magnitude std to DALI bin-index scale."""
    return mstd * (num_bins - 1) / _TIMM_LEVEL_DENOM


# ---------------------------------------------------------------------------
# Augmentation suites (matching timm)
# ---------------------------------------------------------------------------


def get_timm_increasing_suite(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> List[_Augmentation]:
    """Augmentation suite matching timm's ``_RAND_INCREASING_TRANSFORMS``.

    15 operations: AutoContrast, Equalize, Invert, Rotate,
    PosterizeIncreasing, SolarizeIncreasing, SolarizeAdd,
    ColorIncreasing, ContrastIncreasing, BrightnessIncreasing,
    SharpnessIncreasing, ShearX, ShearY, TranslateXRel, TranslateYRel.
    """
    default_translate_abs = 100
    default_translate_rel = 0.45  # timm default translate_pct

    translations = get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )

    return translations + [
        a.shear_x.augmentation((0, 0.3), True),
        a.shear_y.augmentation((0, 0.3), True),
        a.rotate.augmentation((0, 30), True),
        a.brightness.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.contrast.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.color.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.sharpness.augmentation((0, 0.9), True, a.sharpness_kernel),
        # timm PosterizeIncreasing: bits_to_keep goes from 4 (weak) to 0 (strong)
        a.posterize.augmentation((4, 0), False, a.poster_mask_uint8),
        # timm SolarizeIncreasing: threshold goes from 256 (weak) to 0 (strong)
        a.solarize.augmentation((256, 0)),
        # timm SolarizeAdd: shift goes from 0 (weak) to 110 (strong)
        a.solarize_add.augmentation((0, 110), False, a.solarize_add_shift),
        a.equalize,
        a.auto_contrast,
        a.invert,
    ]


def get_timm_default_suite(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> List[_Augmentation]:
    """Augmentation suite matching timm's ``_RAND_TRANSFORMS`` (non-increasing).

    Differences from increasing: Posterize, Solarize, Color, Contrast,
    Brightness, Sharpness use non-monotonic magnitude ranges.
    """
    default_translate_abs = 100
    default_translate_rel = 0.45

    translations = get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )

    return translations + [
        a.shear_x.augmentation((0, 0.3), True),
        a.shear_y.augmentation((0, 0.3), True),
        a.rotate.augmentation((0, 30), True),
        # Non-increasing: enhance ops use (0.1, 1.9) range without shift
        a.brightness.augmentation((0.1, 1.9), False, None),
        a.contrast.augmentation((0.1, 1.9), False, None),
        a.color.augmentation((0.1, 1.9), False, None),
        a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
        # Non-increasing posterize: magnitude (0, 4) means 0 bits removed -> 4 bits removed
        a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
        # Non-increasing solarize: threshold (0, 256) means 0 -> 256
        a.solarize.augmentation((0, 256), False, None),
        a.solarize_add.augmentation((0, 110), False, a.solarize_add_shift),
        a.equalize,
        a.auto_contrast,
        a.invert,
    ]


# ---------------------------------------------------------------------------
# Custom apply_rand_augment with mstd and per-op probability
# ---------------------------------------------------------------------------


def _sample_magnitude_bin(
    m: int,
    mstd: float,
    num_bins: int,
) -> Union[int, _DataNode]:
    """Sample a magnitude bin, optionally adding Gaussian noise.

    When ``mstd == 0``, returns the fixed int ``m``.
    When ``mstd > 0``, returns a per-sample ``DataNode`` of int32 bins drawn
    from ``Normal(m, mstd)`` clamped to ``[0, num_bins - 1]``.
    When ``mstd == inf``, returns a per-sample ``DataNode`` drawn from
    ``Uniform(0, m)``.
    """
    if mstd == 0:
        return m

    if mstd == float("inf"):
        random_m = fn.random.uniform(range=(0.0, float(m)))
    else:
        random_m = fn.random.normal(mean=float(m), stddev=mstd)

    hi = float(num_bins - 1)
    random_m = dali_math.clamp(random_m, 0.0, hi)

    # Round to nearest int, then clamp to valid range to guard against
    # edge cases where cast rounds 30.5 up to 31 instead of truncating.
    rounded = fn.cast(random_m + 0.5, dtype=types.INT32)
    return dali_math.clamp(rounded, 0, num_bins - 1)


def apply_rand_augment_with_mstd(
    augmentations: List[_Augmentation],
    data: _DataNode,
    n: int,
    m: int,
    mstd: float = 0.0,
    p: float = 1.0,
    num_magnitude_bins: int = _DEFAULT_NUM_MAGNITUDE_BINS,
    seed: Optional[int] = None,
    **kwargs,
) -> _DataNode:
    """Apply RandAugment with per-operation magnitude noise and skip probability.

    Like DALI's ``apply_rand_augment`` but supports:

    * ``mstd``: per-operation Gaussian noise on magnitude bin (timm's
      ``magnitude_std``).  When 0, magnitude is fixed.  When > 0, each
      selected operation gets its own magnitude drawn from
      ``Normal(m, mstd)`` clamped to ``[0, num_magnitude_bins - 1]``.
      When ``inf``, drawn from ``Uniform(0, m)``.
    * ``p``: per-operation application probability (timm default 0.5).
      Each of the ``n`` selected operations is independently skipped with
      probability ``1 - p``.
    """
    if n == 0:
        return data
    if len(augmentations) == 0:
        raise ValueError("augmentations list cannot be empty when n > 0")

    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    forbid_unused_kwargs(augmentations, kwargs, "apply_rand_augment_with_mstd")

    # Sample n operation indices per sample (one per level)
    shape = () if n == 1 else (n,)
    op_idx = fn.random.uniform(
        values=list(range(len(augmentations))),
        seed=seed,
        shape=shape,
        dtype=types.INT32,
    )

    for level_idx in range(n):
        # Per-level magnitude with optional noise
        level_m = _sample_magnitude_bin(m, mstd, num_magnitude_bins)

        if use_signed_magnitudes:
            mag_bin = signed_bin(level_m, seed=seed)
        else:
            mag_bin = level_m

        level_op_idx = op_idx if n == 1 else op_idx[level_idx]

        op_kwargs = dict(
            data=data,
            magnitude_bin=mag_bin,
            num_magnitude_bins=num_magnitude_bins,
            **kwargs,
        )

        if p < 1.0:
            should_apply = fn.random.coin_flip(probability=p)
            if should_apply:
                data = pretty_select(
                    augmentations,
                    level_op_idx,
                    op_kwargs,
                    auto_aug_name="apply_rand_augment_with_mstd",
                    ref_suite_name="get_timm_increasing_suite",
                )
        else:
            data = pretty_select(
                augmentations,
                level_op_idx,
                op_kwargs,
                auto_aug_name="apply_rand_augment_with_mstd",
                ref_suite_name="get_timm_increasing_suite",
            )

    return data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def dali_rand_augment(
    data: _DataNode,
    config_str: str,
    shape: Optional[Union[_DataNode, Tuple[int, int]]] = None,
    fill_value: int = 128,
    interp_type: Optional[types.DALIInterpType] = None,
    num_magnitude_bins: int = _DEFAULT_NUM_MAGNITUDE_BINS,
    seed: Optional[int] = None,
) -> _DataNode:
    """Apply RandAugment to a batch of uint8 HWC images, matching timm.

    Args:
        data: batch of uint8 HWC images.
        config_str: timm config string, e.g. ``'rand-m9-mstd0.5-inc1'``.
        shape: image shapes (H, W) for relative translation magnitudes.
            Can be a ``DataNode`` from ``fn.peek_image_shape`` or a fixed
            ``(H, W)`` tuple.  When ``None``, absolute pixel offsets are used
            (max 100 px, matching timm's ``translate_const=250`` rescaled to
            DALI's default range).
        fill_value: padding value for affine transforms.
        interp_type: interpolation for affine transforms.
        num_magnitude_bins: number of discrete magnitude bins.
        seed: random seed.

    Returns:
        Augmented batch of uint8 HWC images.
    """
    cfg = parse_rand_augment_config(config_str)

    m_dali = _timm_m_to_dali(cfg["m"], num_magnitude_bins)
    mstd_dali = _timm_mstd_to_dali(cfg["mstd"], num_magnitude_bins) if 0 < cfg["mstd"] < float("inf") else cfg["mstd"]

    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape

    if cfg["inc"]:
        augmentations = get_timm_increasing_suite(
            use_shape=use_shape,
            max_translate_rel=0.45 if use_shape else None,
            max_translate_abs=100 if not use_shape else None,
        )
    else:
        augmentations = get_timm_default_suite(
            use_shape=use_shape,
            max_translate_rel=0.45 if use_shape else None,
            max_translate_abs=100 if not use_shape else None,
        )

    return apply_rand_augment_with_mstd(
        augmentations,
        data,
        n=cfg["n"],
        m=m_dali,
        mstd=mstd_dali,
        p=cfg["p"],
        num_magnitude_bins=num_magnitude_bins,
        seed=seed,
        **aug_kwargs,
    )
