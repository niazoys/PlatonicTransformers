import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Native PyTorch scatter ops — torch.compile compatible, no torch_scatter dependency.
def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: int = None) -> torch.Tensor:
    """Native PyTorch replacement for torch_scatter.scatter_sum."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    return out.scatter_add_(dim, idx, src)


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                     dim_size: int = None) -> torch.Tensor:
    """Native PyTorch replacement for torch_scatter.scatter_softmax."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    max_vals = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    src_shifted = src - max_vals[index]
    exp_src = torch.exp(src_shifted)
    sum_exp = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    sum_exp.scatter_add_(0, index, exp_src)
    return exp_src / (sum_exp[index] + 1e-16)

try:
    from torch_cluster import knn_graph
except (ImportError, OSError):
    knn_graph = None

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except (ImportError, OSError):  # flash-attn is optional
    flash_attn_varlen_func = None

from platonic_transformers.models.platoformer.utils import scatter_add
from platonic_transformers.models.platoformer.rope import PlatonicRoPE
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS


class _HeadDimRMSNorm(nn.Module):
    """RMSNorm along the last (head_dim) axis. Manual implementation because
    torch.nn.RMSNorm triggers hundreds of per-shape specializations under
    torch.compile + dynamic shapes, whereas a plain tensor op traces cleanly.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class PlatonicConv(nn.Module):
    """
    Computes a group-equivariant dynamic convolution supporting both graph and dense modes.

    This layer uses Rotary Positional Embeddings (RoPE) to compute a dynamic
    convolution kernel. It supports two modes for dense data:
    1.  attention=False (Default): A highly efficient linear convolutio type "attention" mechanism.
    2.  attention=True: Equivariant scaled dot-product attention with softmax.
    
    Graph-structured data only uses the linear attention mechanism.
    The layer is equivariant to the symmetries of a specified Platonic solid.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        freq_init: str = 'random',
        learned_freqs: bool = True,
        bias: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
        rope_on_values: bool = False,
        attention_backend: str = "scatter",
        qk_norm: bool = False,
    ):
        super().__init__()

        # --- Group Setup ---
        self.rope_on_values = rope_on_values
        if attention_backend not in ("scatter", "flash"):
            raise ValueError(
                f"attention_backend must be 'scatter' or 'flash', got {attention_backend!r}"
            )
        if attention_backend == "flash" and flash_attn_varlen_func is None:
            raise ImportError(
                "attention_backend='flash' requires the flash-attn package "
                "(pip install flash-attn)."
            )
        self.attention_backend = attention_backend
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        
        # --- Dimension Validation and Setup ---
        if in_channels % self.num_G != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by group size ({self.num_G}).")
        self.in_channels_g = in_channels // self.num_G

        if out_channels % self.num_G != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by group size ({self.num_G}).")
        self.out_channels_g = out_channels // self.num_G

        if num_heads % self.num_G != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by group size ({self.num_G}).")

        if embed_dim % (num_heads // self.num_G) != 0:
             raise ValueError(f"embed_dim ({embed_dim}) must be divisible by (num_heads // num_group) = "
                             f"{self.num_G // num_heads}.")
        self.embed_dim = embed_dim
        self.embed_dim_g = embed_dim // self.num_G
       
        self.out_channels = out_channels
        self.effective_num_heads = num_heads//self.num_G
        self.head_dim = self.embed_dim_g // self.effective_num_heads

        self.mean_aggregation = mean_aggregation
        self.attention = attention

        # --- Sub-modules ---
        self.use_key = use_key
        self.q_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        self.v_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        if freq_sigma is None or use_key:
            self.k_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        else:
            self.register_buffer('k_proj', None)

        # Group-equivariant RoPE for positional information
        if freq_sigma is not None:
            self.rope_emb = PlatonicRoPE(
                embed_dim=embed_dim,
                num_heads=self.effective_num_heads,
                head_dim=self.head_dim,
                solid_name=solid_name,
                spatial_dims=spatial_dims,
                freq_sigma=freq_sigma,
                learned_freqs=learned_freqs,
                freq_init=freq_init
            )
        else:
            self.register_buffer('rope_emb', None)

        # Final equivariant linear layer
        self.out_proj = PlatonicLinear(embed_dim, out_channels, solid_name, bias=bias)

        # Optional QK-norm: RMSNorm along head_dim on Q and K, applied before
        # RoPE / attention. Bounds attention scores regardless of how large
        # the underlying activations get — used in SD3, FLUX, HunyuanDiT,
        # ViT-22B to prevent attention-saturation / softmax-collapse
        # instability at scale. Equivariance-preserving: the normalization is
        # along head_dim, which the group action does not permute
        # (the group permutes G and H axes; head_dim is invariant).
        #
        # Manual implementation instead of nn.RMSNorm because the latter
        # triggers hundreds of per-shape specializations under torch.compile
        # + dynamic batch shapes (our QM9 molecules have variable atom
        # counts). A plain tensor op traces cleanly with zero recompiles.
        if qk_norm:
            self.q_norm = _HeadDimRMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = _HeadDimRMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
    
    def _forward_shared(self, x: Tensor, pos: Tensor):
        """Shared logic for projections and RoPE application."""
        leading_dims = x.shape[:-1]
        
        q_raw = self.q_proj(x)
        v_raw = self.v_proj(x)
        # If not using RoPE, then project, but if using RoPE then use ones
        k_raw = self.k_proj(x) if ((self.rope_emb is None) or self.use_key) else torch.ones_like(q_raw)

        # Reshape for multi-head processing: [..., G * H * D_h] -> [..., G, H, D_h]
        q = q_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)
        v = v_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)
        k = k_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)

        # QK-norm (identity if disabled) before RoPE. Normalizes each
        # (N, G, H) head-vector along head_dim, so attention scores q·k are
        # bounded regardless of input activation magnitude.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to query and key (and optionally value, per GTA Eq. 5)
        if self.rope_emb is not None:
            q = self.rope_emb(q, pos)
            k = self.rope_emb(k, pos)
            if self.rope_on_values:
                v = self.rope_emb(v, pos)

        return q, k, v

    def graph_scattered_attention(self,
        q: torch.Tensor,      # [N, G, H, D]
        k: torch.Tensor,      # [N, G, H, D]
        v: torch.Tensor,      # [N, G, H, D]
        batch: torch.Tensor,  # [N]
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        k_knn: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute fully connected edges if edge_index is None, or kNN edges if k_knn is given.
        Uses an equivariant attention mechanism over the combined group/head dimension.

        Returns
        -------
        out : Tensor, shape [N, G*H*D]
        """
        N, G, H, D = q.shape
        device = q.device

        if edge_index is not None:
            src, dst = edge_index.to(device)
        elif k_knn is not None:
            if pos is None:
                raise ValueError("k_knn was given but 'pos' is None.")
            if knn_graph is None:
                raise ImportError("torch_cluster.knn_graph is required for kNN mode.")
            edge_index = knn_graph(
                x=pos.to(device), k=k_knn, batch=batch, loop=True
            )                                        # [2, |E_knn|]
            src, dst = edge_index
        else:
            N = batch.shape[0]
            # Create a dense N x N grid of all possible edges
            node_idx = torch.arange(N, device=device)
            src, dst = torch.meshgrid(node_idx, node_idx, indexing='ij')

            # Keep only the edges where the source and destination nodes
            # belong to the same graph in the batch.
            mask = batch[src] == batch[dst]
            
            # Apply the mask to get the final edge index
            src = src[mask]
            dst = dst[mask]

        E = src.numel()
        
        GH = G * H
        q_src = q.reshape(N, GH, D)[src]  # [E, GH, D]
        k_dst = k.reshape(N, GH, D)[dst]  # [E, GH, D]
        v_dst = v.reshape(N, GH, D)[dst]  # [E, GH, D]

        scores = (q_src * k_dst).sum(-1) * D ** -0.5  # [E, GH]

        # reindex ids for heads
        head_ids = torch.arange(GH, device=device).repeat(E, 1)     # [E, GH]
        group_ids = src.unsqueeze(1) * GH + head_ids                # [E, GH]

        a = _scatter_softmax(
            scores.flatten(),
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(E, GH)

        weighted = (a.unsqueeze(-1) * v_dst).reshape(-1, D)         # [E*GH, D]

        out = _scatter_sum(
            weighted,
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(N, GH, D)

        # Reshape (N, GH, D) -> (N, G*H*D)
        return out.reshape(N, G * H * D)
        


    def graph_flash_varlen_attention(self,
        q: torch.Tensor,      # [N, G, H, D]
        k: torch.Tensor,      # [N, G, H, D]
        v: torch.Tensor,      # [N, G, H, D]
        batch: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Equivalent to graph_scattered_attention (fully-connected within-graph)
        but computed via flash_attn_varlen_func — one fused CUDA kernel, no
        explicit N x N meshgrid/mask.

        Parameters
        ----------
        q, k, v : [N, G, H, D]  query, key, value tensors (post-RoPE).
        batch   : [N]  per-node graph index (values 0..B-1, sorted by graph).

        Returns
        -------
        out : [N, G*H*D]

        Equivariance: the G axis is treated as an extra head dimension.
        Because the group acts on the G axis by permutation, and flash varlen
        applies independent attention per head, the operation commutes with
        the group action -> equivariance preserved (matches the scatter
        implementation semantics).

        Dtype: flash_attn_varlen_func requires fp16/bf16. We cast q,k,v to
        bf16 inside the kernel call and cast the output back to the input
        dtype. For fp32 callers this means a ~bf16 level of rounding error
        inside attention but the residual path remains fp32.
        """
        N, G, H, D = q.shape
        GH = G * H
        device = q.device

        # Contiguous [N, GH, D] view expected by flash varlen.
        q_flat = q.reshape(N, GH, D).contiguous()
        k_flat = k.reshape(N, GH, D).contiguous()
        v_flat = v.reshape(N, GH, D).contiguous()

        # Cumulative sequence lengths [B+1] and max seq length in this batch.
        # We assume `batch` is sorted (standard PyG convention), so counts can
        # be derived by one bincount pass.
        B = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=B).to(dtype=torch.int32)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(counts, dim=0)
        max_seqlen = int(counts.max().item())

        # Flash Attention requires fp16 / bf16. Cast before call, cast back after.
        orig_dtype = q_flat.dtype
        q_bf = q_flat.to(torch.bfloat16)
        k_bf = k_flat.to(torch.bfloat16)
        v_bf = v_flat.to(torch.bfloat16)

        out_bf = flash_attn_varlen_func(
            q_bf, k_bf, v_bf,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )  # [N, GH, D]

        out = out_bf.to(orig_dtype)
        return out.reshape(N, GH * D)

    def _forward_graph(self, x: Tensor, pos: Tensor, batch: Tensor, avg_num_nodes=1.0):
        """
        Implementation for graph-structured data.
        Supports both kernelized linear attention and standard softmax attention.
        """
        q_rope, k_rope, v = self._forward_shared(x, pos) # [N, G, H, D_h]

        if self.attention:
            if self.attention_backend == "flash":
                output = self.graph_flash_varlen_attention(q_rope, k_rope, v, batch)
            else:
                output = self.graph_scattered_attention(q_rope, k_rope, v, batch, pos)
            # Un-rotate output (GTA Eq. 5: O_i = ρ(g_i)⁻¹ * weighted_sum)
            if self.rope_on_values and self.rope_emb is not None:
                output_ghd = output.view(-1, self.num_G, self.effective_num_heads, self.head_dim)
                output_ghd = self.rope_emb(output_ghd, pos, inverse=True)
                output = output_ghd.flatten(-3, -1)
        else:
            kv_outer_product = torch.einsum('nghd,nghe->nghde', k_rope, v)
            num_graphs = batch.max() + 1
            kv_kernel = scatter_add(kv_outer_product, batch, dim_size=num_graphs)

            if self.mean_aggregation:
                num_nodes = scatter_add(torch.ones_like(batch, dtype=torch.float), batch, dim_size=num_graphs)[..., None, None, None, None]
            else:
                num_nodes = avg_num_nodes
            kv_kernel = kv_kernel / num_nodes

            output = torch.einsum('nghd,nghde->nghe', q_rope, kv_kernel[batch])
            # Un-rotate output for linear attention
            if self.rope_on_values and self.rope_emb is not None:
                output = self.rope_emb(output, pos, inverse=True)
            output = output.flatten(-3, -1) # -> (..., G, H, H_dim) -> (..., G*H*H_dim)

        return self.out_proj(output)

    def _forward_dense(self, x: Tensor, pos: Tensor, mask: Tensor, avg_num_nodes=1.0):
        """
        Implementation for dense, padded data.
        Supports both linear and standard softmax attention.
        """
        q_rope, k_rope, v = self._forward_shared(x, pos)
        B, S, _ = x.shape # B: batch size, S: sequence length

        if self.attention:
            # Reshape for scaled_dot_product_attention: (B, S, G, H, Dh) -> (B, G*H, S, Dh)
            q_sdpa = q_rope.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)
            k_sdpa = k_rope.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)
            v_sdpa = v.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)

            attn_mask = mask[:, None, None, :] if mask is not None else None
            # Let PyTorch auto-select the fastest SDPA backend (CuDNN on H100, Flash-2 elsewhere)
            attn_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

            # Reshape back: (B, G*H, S, Dh) -> (B, S, G, H, Dh)
            attn_output = attn_output.transpose(1, 2).view(B, S, self.num_G, self.effective_num_heads, self.head_dim)
            # Un-rotate output (GTA Eq. 5: O_i = ρ(g_i)⁻¹ * weighted_sum)
            if self.rope_on_values and self.rope_emb is not None:
                attn_output = self.rope_emb(attn_output, pos, inverse=True)
            output = attn_output.reshape(B, S, self.embed_dim)
        else:
            if mask is not None:
                # Apply mask before aggregation
                v = v * mask[..., None, None, None]
                k_rope = k_rope * mask[..., None, None, None]

            kv_kernel = torch.einsum('bsghd,bsghe->bghde', k_rope, v)

            if self.mean_aggregation and mask is not None:
                num_nodes = mask.sum(dim=-1).float().view(B, 1, 1, 1, 1)
            else:
                num_nodes = avg_num_nodes
            kv_kernel = kv_kernel / num_nodes

            output = torch.einsum('bsghd,bghde->bsghe', q_rope, kv_kernel)
            output = output.flatten(-3, -1)
        
        return self.out_proj(output)
    
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        avg_num_nodes: Optional[float] = 1.0
    ) -> Tensor:
        is_graph_mode = batch is not None
        avg_num_nodes = avg_num_nodes if avg_num_nodes is not None else 1.0
        
        if is_graph_mode:
            if mask is not None:
                raise ValueError("Only one of 'batch' or 'mask' can be provided.")
            return self._forward_graph(x, pos, batch, avg_num_nodes=avg_num_nodes)
        else:
            return self._forward_dense(x, pos, mask, avg_num_nodes=avg_num_nodes)
