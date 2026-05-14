"""Shared optimizer helpers for the dataset-specific mains.

Centralizes the param-group split for weight decay so every main applies the
same convention. Standard idiom (ViT / ConvNeXt / GPT / DeiT-III): apply
weight decay only to weight matrices and skip:

  - 1D tensors: biases, LayerNorm gamma/beta, LayerScale gammas. Decaying
    these fights normalization and shrinks the residual stream.
  - APE/RoPE learnable frequency parameters (named ``freqs``). Decay degrades
    the position encoding by pulling frequencies toward zero.
  - ``nn.Embedding`` weights, by convention.

Use ``make_param_groups(model, weight_decay)`` and pass the returned list to
any torch optimizer (Adam / AdamW / Lamb / ...).
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch import nn


def make_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Split ``model``'s trainable parameters into decay / no-decay groups."""
    embedding_param_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            for p in m.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_freq = name.endswith(".freqs") or name == "freqs"
        is_embedding = id(p) in embedding_param_ids
        if p.dim() < 2 or name.endswith(".bias") or is_freq or is_embedding:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups
