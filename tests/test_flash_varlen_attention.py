"""Tests for the flash-attn varlen replacement of graph_scattered_attention.

Two things to verify when swapping in `attention_backend="flash"`:

1. **Numerical equivalence to scatter**, within the precision cost of casting
   q,k,v to bf16 inside the kernel (we expect ~1e-2 absolute / 1e-3 relative
   on a small model with random inputs — far better than full bf16 training,
   because only the attention matmul is bf16 here).

2. **Equivariance is preserved** under the Platonic group action. The flash
   path treats the (G, H) head dim as a single H_total axis, and the group
   acts on G by permutation. Since flash applies independent attention per
   head, the operation commutes with the group action.

These tests require a GPU (flash-attn is a CUDA-only kernel). They are
skipped in CPU-only environments.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

flash_attn_varlen_func = pytest.importorskip("flash_attn").flash_attn_varlen_func

if not torch.cuda.is_available():
    pytest.skip("flash-attn requires CUDA; skipping all tests in this module.",
                allow_module_level=True)

from platonic_transformers.models.platoformer.conv import PlatonicConv
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_inputs(N: int, B: int, d_model: int, device: str = "cuda"):
    """Make a sparse batched input. Each graph i has N/B nodes (allowing the
    last graph to absorb the remainder)."""
    sizes = [N // B] * B
    sizes[-1] += N - sum(sizes)
    batch = torch.cat([
        torch.full((sizes[i],), i, dtype=torch.long, device=device)
        for i in range(B)
    ])
    x = torch.randn(N, d_model, device=device)
    pos = torch.randn(N, 3, device=device)
    return x, pos, batch


def _build_conv(solid_name: str, d_model: int, nhead: int, backend: str, device: str = "cuda"):
    G = PLATONIC_GROUPS[solid_name].G
    return PlatonicConv(
        in_channels=d_model,
        out_channels=d_model,
        embed_dim=d_model,
        num_heads=nhead,
        solid_name=solid_name,
        spatial_dims=3,
        freq_sigma=2.0,
        learned_freqs=True,
        attention=True,
        rope_on_values=True,
        attention_backend=backend,
    ).to(device)


@pytest.mark.parametrize("solid_name", ["tetrahedron", "octahedron"])
def test_scatter_vs_flash_numerical(solid_name):
    """Outputs must match within bf16 tolerance (~1e-2 abs)."""
    G = PLATONIC_GROUPS[solid_name].G
    d_model = 12 * G  # divisible by G and by nhead
    nhead = G

    scatter = _build_conv(solid_name, d_model, nhead, "scatter")
    flash = _build_conv(solid_name, d_model, nhead, "flash")
    # Tie weights so the only thing being compared is the attention path.
    flash.load_state_dict(scatter.state_dict())

    x, pos, batch = _make_inputs(N=12, B=3, d_model=d_model)

    with torch.no_grad():
        out_scatter = scatter(x, pos, batch=batch)
        out_flash = flash(x, pos, batch=batch)

    abs_err = (out_scatter - out_flash).abs().max().item()
    rel = (out_scatter - out_flash).norm() / out_scatter.norm().clamp_min(1e-8)
    print(f"{solid_name}: max_abs={abs_err:.3e}  rel_l2={rel.item():.3e}")
    # bf16 has ~7 mantissa bits -> relative error ~1e-2 on chained ops.
    assert abs_err < 5e-2, f"flash output diverges from scatter: max_abs={abs_err}"
    assert rel.item() < 5e-2, f"flash output diverges from scatter: rel_l2={rel.item()}"


@pytest.mark.parametrize("solid_name", ["tetrahedron", "octahedron"])
def test_flash_equivariance(solid_name):
    """Group action commutes with flash attention path."""
    G = PLATONIC_GROUPS[solid_name]
    d_model = 12 * G.G
    nhead = G.G

    model = PlatonicTransformer(
        input_dim=5, input_dim_vec=0, hidden_dim=d_model,
        output_dim=5, output_dim_vec=1,
        nhead=nhead, num_layers=2, solid_name=solid_name,
        scalar_task_level="node", vector_task_level="node",
        attention=True, rope_sigma=2.0, freq_init="random",
        rope_on_values=True, attention_backend="flash",
        time_conditioning=True,
    ).cuda().eval()

    # Nudge weights off zero-init so the modulation actually fires.
    for p in model.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p) * 0.02)

    N, B = 12, 2
    x = torch.randn(N, 5, device="cuda")
    pos = torch.randn(N, 3, device="cuda")
    batch = torch.tensor([0] * 6 + [1] * 6, device="cuda")
    t = torch.randn(B, device="cuda")

    R = G.elements[3].float().cuda()
    R_nodes = R.unsqueeze(0).expand(N, -1, -1)

    with torch.no_grad():
        s0, v0 = model(x, pos, batch=batch, t=t)
        pos_rot = torch.einsum("nij,nj->ni", R_nodes, pos)
        s1, v1 = model(x, pos_rot, batch=batch, t=t)

    v0_rot = torch.einsum("nij,nkj->nki", R_nodes, v0)
    s_err = (s1 - s0).abs().max().item()
    v_err = (v1 - v0_rot).abs().max().item()
    print(f"{solid_name}: scalar_err={s_err:.3e}  vector_err={v_err:.3e}")
    # bf16 attention -> looser threshold than the fp32 scatter test (1e-5).
    assert s_err < 1e-2, f"scalar invariance broken: {s_err}"
    assert v_err < 1e-2, f"vector covariance broken: {v_err}"
