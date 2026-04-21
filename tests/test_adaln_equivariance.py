"""Equivariance tests for the AdaLN (diffusion-conditioning) path.

These verify that diffusion-style time conditioning does not break the Platonic
symmetry: shift/scale/gate are shared across the group axis, so for any group
element R the model's scalar outputs are invariant and its vector outputs are
covariant under R.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.block import PlatonicBlock
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


@pytest.fixture(autouse=True)
def _determinism():
    torch.manual_seed(0)


def _nudge(model: torch.nn.Module, scale: float = 0.02) -> None:
    """Break zero-init so the AdaLN path actually exercises shift/scale/gate."""
    for p in model.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p) * scale)


@pytest.mark.parametrize("solid_name", ["tetrahedron", "octahedron"])
def test_block_identity_at_init(solid_name):
    G = PLATONIC_GROUPS[solid_name].G
    d_model = 12 * G  # divisible by G and by nhead
    block = PlatonicBlock(
        d_model=d_model, nhead=G, dim_feedforward=2 * d_model,
        solid_name=solid_name, conditioning_dim=64, attention=True, dropout=0.0,
    )
    N, B = 8, 2
    x = torch.randn(N, d_model)
    pos = torch.randn(N, 3)
    batch = torch.tensor([0] * 4 + [1] * 4)
    cond = torch.randn(B, 64)

    with torch.no_grad():
        out = block(x, pos, batch=batch, conditioning=cond)
    # Zero-init: gate=0 -> residual is fully suppressed -> block is identity.
    assert torch.allclose(out, x, atol=0, rtol=0)


@pytest.mark.parametrize("solid_name", ["tetrahedron", "octahedron"])
def test_transformer_equivariance_with_time_conditioning(solid_name):
    G = PLATONIC_GROUPS[solid_name]
    d_model = 12 * G.G
    model = PlatonicTransformer(
        input_dim=5, input_dim_vec=0, hidden_dim=d_model,
        output_dim=5, output_dim_vec=1,
        nhead=G.G, num_layers=3, solid_name=solid_name,
        scalar_task_level="node", vector_task_level="node",
        attention=True, rope_sigma=2.0, freq_init="random",
        rope_on_values=True, time_conditioning=True,
    ).eval()
    _nudge(model)

    N, B = 10, 2
    x = torch.randn(N, 5)
    pos = torch.randn(N, 3)
    batch = torch.tensor([0] * 5 + [1] * 5)
    t = torch.randn(B)

    R = G.elements[3].float()  # pick a non-identity group element
    R_batch = R.unsqueeze(0).expand(N, -1, -1)

    with torch.no_grad():
        s0, v0 = model(x, pos, batch=batch, t=t)
        pos_rot = torch.einsum("nij,nj->ni", R_batch, pos)
        s1, v1 = model(x, pos_rot, batch=batch, t=t)

    v0_rot = torch.einsum("nij,nkj->nki", R_batch, v0)
    assert (s1 - s0).abs().max().item() < 1e-5, "scalar outputs must be invariant"
    assert (v1 - v0_rot).abs().max().item() < 1e-5, "vector outputs must be covariant"


def test_backward_reaches_adaLN():
    """Sanity: diffusion conditioning gradients actually flow into adaLN_modulation."""
    model = PlatonicTransformer(
        input_dim=5, input_dim_vec=0, hidden_dim=144,
        output_dim=5, output_dim_vec=1,
        nhead=12, num_layers=2, solid_name="tetrahedron",
        scalar_task_level="node", vector_task_level="node",
        attention=True, time_conditioning=True,
    )
    N, B = 6, 2
    x = torch.randn(N, 5)
    pos = torch.randn(N, 3)
    batch = torch.tensor([0] * 3 + [1] * 3)
    t = torch.randn(B, requires_grad=False)

    s, v = model(x, pos, batch=batch, t=t)
    (s.sum() + v.sum()).backward()
    g = model.layers[0].adaLN_modulation[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "AdaLN weights must receive gradients"
