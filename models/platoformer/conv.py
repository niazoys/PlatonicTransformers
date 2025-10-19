import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import Tensor
from typing import Optional
import math

from torch_scatter import scatter_softmax,scatter_sum

try:
    from torch_cluster import knn_graph  
except ImportError:
    knn_graph = None

# Assumes these modules are in your project structure
from .utils import scatter_add
from .rope import PlatonicRoPE
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS


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
        use_key: bool = False
    ):
        super().__init__()

        # --- Group Setup ---
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

        # Apply RoPE to query and key
        if self.rope_emb is not None:
            q = self.rope_emb(q, pos)
            k = self.rope_emb(k, pos)

        return q, k, v

    def graph_scattered_attention(self,
        q: torch.Tensor,      # [N, G, H, D]
        k: torch.Tensor,      # [N, G, H, D]
        v: torch.Tensor,      # [N, G, H, D]
        batch: torch.Tensor,  # [N]
        pos: torch.Tensor | None = None,     
        edge_index: torch.Tensor | None = None,
        k_knn: int | None = None
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

        a = scatter_softmax(
            scores.flatten(),
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(E, GH)

        weighted = (a.unsqueeze(-1) * v_dst).reshape(-1, D)         # [E*GH, D]

        out = scatter_sum(
            weighted,
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(N, GH, D)

        # Reshape (N, GH, D) -> (N, G*H*D)
        return out.reshape(N, G * H * D)
        


    def _forward_graph(self, x: Tensor, pos: Tensor, batch: Tensor, avg_num_nodes=1.0):
        """
        Implementation for graph-structured data.
        Supports both kernelized linear attention and standard softmax attention.
        """
        q_rope, k_rope, v = self._forward_shared(x, pos) # [N, G, H, D_h]
  
        if self.attention:
            output = self.graph_scattered_attention(q_rope, k_rope, v, batch, pos)
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
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):      
               attn_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

            # Reshape back for output projection: (B, G*H, S, Dh)/(B, H, S, G*Dh) -> (B, S, G*H*Dh)
            output = attn_output.transpose(1, 2).reshape(B, S, self.embed_dim)
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
