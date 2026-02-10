"""
Group-equivariant convolution layer for JAX/Flax.

Implements PlatonicConv which computes group-equivariant dynamic convolution
with support for both graph and dense modes.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer_jax.linear import PlatonicLinear
from platonic_transformers.models.platoformer_jax.rope import PlatonicRoPE
from platonic_transformers.models.platoformer_jax.utils import segment_sum


class PlatonicConv(nn.Module):
    """
    Group-equivariant dynamic convolution layer.
    
    Uses Rotary Positional Embeddings (RoPE) to compute dynamic convolution kernels.
    Supports both linear attention (default) and standard softmax attention.
    
    Attributes:
        in_channels: Number of input channels (must be divisible by G)
        out_channels: Number of output channels (must be divisible by G)
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        solid_name: Name of the Platonic solid defining the symmetry group
        spatial_dims: Number of spatial dimensions
        freq_sigma: Standard deviation for RoPE frequencies
        freq_init: Initialization method for frequencies
        learned_freqs: If True, frequencies are learnable
        use_bias: Whether to use bias in projections
        mean_aggregation: Whether to normalize by number of nodes
        attention: If True, use standard softmax attention
        use_key: If True, learn separate key projection
    """
    in_channels: int
    out_channels: int
    embed_dim: int
    num_heads: int
    solid_name: str
    spatial_dims: int = 3
    freq_sigma: Optional[float] = 1.0
    freq_init: str = 'spiral'
    learned_freqs: bool = True
    use_bias: bool = True
    mean_aggregation: bool = False
    attention: bool = False
    use_key: bool = False
    
    def setup(self):
        """Initialize the layer."""
        self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        self.num_G = self.group.G
        
        # Validate dimensions
        if self.in_channels % self.num_G != 0:
            raise ValueError(f"in_channels ({self.in_channels}) must be divisible by group size ({self.num_G}).")
        self.in_channels_g = self.in_channels // self.num_G
        
        if self.out_channels % self.num_G != 0:
            raise ValueError(f"out_channels ({self.out_channels}) must be divisible by group size ({self.num_G}).")
        self.out_channels_g = self.out_channels // self.num_G
        
        if self.num_heads % self.num_G != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by group size ({self.num_G}).")
        
        self.effective_num_heads = self.num_heads // self.num_G
        self.embed_dim_g = self.embed_dim // self.num_G
        self.head_dim = self.embed_dim_g // self.effective_num_heads
        
        # Projections
        self.q_proj = PlatonicLinear(self.in_channels, self.embed_dim, self.solid_name, use_bias=self.use_bias)
        self.v_proj = PlatonicLinear(self.in_channels, self.embed_dim, self.solid_name, use_bias=self.use_bias)
        
        if self.freq_sigma is None or self.use_key:
            self.k_proj = PlatonicLinear(self.in_channels, self.embed_dim, self.solid_name, use_bias=self.use_bias)
            self._use_k_proj = True
        else:
            self._use_k_proj = False
        
        # RoPE for positional information
        if self.freq_sigma is not None:
            self.rope_emb = PlatonicRoPE(
                embed_dim=self.embed_dim,
                num_heads=self.effective_num_heads,
                head_dim=self.head_dim,
                solid_name=self.solid_name,
                spatial_dims=self.spatial_dims,
                freq_sigma=self.freq_sigma,
                learned_freqs=self.learned_freqs,
                freq_init=self.freq_init
            )
            self._use_rope = True
        else:
            self._use_rope = False
        
        # Output projection
        self.out_proj = PlatonicLinear(self.embed_dim, self.out_channels, self.solid_name, use_bias=self.use_bias)
    
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        batch: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        avg_num_nodes: float = 1.0,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass for the convolution layer.
        
        Args:
            x: Input features of shape [N, in_channels] (graph) or [B, S, in_channels] (dense)
            pos: Positions of shape [N, spatial_dims] (graph) or [B, S, spatial_dims] (dense)
            batch: Batch indices of shape [N] (only for graph mode)
            mask: Attention mask of shape [B, S] (only for dense mode)
            avg_num_nodes: Average number of nodes for normalization
            deterministic: Whether in inference mode
            
        Returns:
            Output features of same shape as input
        """
        is_graph_mode = batch is not None
        
        if is_graph_mode:
            if mask is not None:
                raise ValueError("Only one of 'batch' or 'mask' can be provided.")
            return self._forward_graph(x, pos, batch, avg_num_nodes)
        else:
            return self._forward_dense(x, pos, mask, avg_num_nodes, deterministic)
    
    def _forward_shared(self, x: jnp.ndarray, pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Shared logic for projections and RoPE application."""
        leading_dims = x.shape[:-1]
        
        q_raw = self.q_proj(x)
        v_raw = self.v_proj(x)
        
        if self._use_k_proj:
            k_raw = self.k_proj(x)
        else:
            k_raw = jnp.ones_like(q_raw)
        
        # Reshape for multi-head: [..., G * H * D_h] -> [..., G, H, D_h]
        q = q_raw.reshape(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)
        v = v_raw.reshape(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)
        k = k_raw.reshape(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim)
        
        # Apply RoPE
        if self._use_rope:
            q = self.rope_emb(q, pos)
            k = self.rope_emb(k, pos)
        
        return q, k, v
    
    def _forward_graph(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        batch: jnp.ndarray,
        avg_num_nodes: float
    ) -> jnp.ndarray:
        """Implementation for graph-structured data using linear attention."""
        q_rope, k_rope, v = self._forward_shared(x, pos)  # [N, G, H, D_h]
        
        # Linear attention: kernel aggregation
        # kv_outer_product: [N, G, H, D, D]
        kv_outer_product = jnp.einsum('nghd,nghe->nghde', k_rope, v)
        
        # Aggregate over graphs
        num_graphs = batch.max() + 1
        kv_kernel = segment_sum(kv_outer_product, batch, num_graphs)
        
        # Normalize
        if self.mean_aggregation:
            num_nodes = segment_sum(jnp.ones_like(batch, dtype=jnp.float32), batch, num_graphs)
            num_nodes = num_nodes.reshape(-1, 1, 1, 1, 1)
        else:
            num_nodes = avg_num_nodes
        kv_kernel = kv_kernel / num_nodes
        
        # Apply query
        output = jnp.einsum('nghd,nghde->nghe', q_rope, kv_kernel[batch])
        output = output.reshape(x.shape[0], -1)  # [N, G*H*D_h]
        
        return self.out_proj(output)
    
    def _forward_dense(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        avg_num_nodes: float,
        deterministic: bool
    ) -> jnp.ndarray:
        """Implementation for dense, padded data."""
        q_rope, k_rope, v = self._forward_shared(x, pos)
        B, S, _ = x.shape
        
        if self.attention:
            # Standard scaled dot-product attention
            # Reshape: (B, S, G, H, Dh) -> (B, G*H, S, Dh)
            q_sdpa = q_rope.reshape(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_sdpa = k_rope.reshape(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_sdpa = v.reshape(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            # Compute attention scores
            scale = 1.0 / jnp.sqrt(self.head_dim)
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q_sdpa, k_sdpa) * scale
            
            # Apply mask if provided
            if mask is not None:
                # mask: [B, S] -> [B, 1, 1, S]
                mask_expanded = mask[:, None, None, :]
                attn_weights = jnp.where(mask_expanded, attn_weights, jnp.finfo(attn_weights.dtype).min)
            
            attn_probs = jax.nn.softmax(attn_weights, axis=-1)
            
            # Apply attention to values
            attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v_sdpa)
            
            # Reshape back: (B, G*H, S, Dh) -> (B, S, G*H*Dh)
            output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
        else:
            # Linear attention
            if mask is not None:
                # Apply mask before aggregation
                v = v * mask[..., None, None, None]
                k_rope = k_rope * mask[..., None, None, None]
            
            # Compute kernel: [B, G, H, D, D]
            kv_kernel = jnp.einsum('bsghd,bsghe->bghde', k_rope, v)
            
            # Normalize
            if self.mean_aggregation and mask is not None:
                num_nodes = mask.sum(axis=-1).astype(jnp.float32).reshape(B, 1, 1, 1, 1)
            else:
                num_nodes = avg_num_nodes
            kv_kernel = kv_kernel / num_nodes
            
            # Apply query
            output = jnp.einsum('bsghd,bghde->bsghe', q_rope, kv_kernel)
            output = output.reshape(B, S, -1)
        
        return self.out_proj(output)
