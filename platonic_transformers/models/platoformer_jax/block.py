"""
Group-equivariant Transformer block for JAX/Flax.

Implements PlatonicBlock which is a Transformer-style block using Platonic symmetries.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer_jax.conv import PlatonicConv
from platonic_transformers.models.platoformer_jax.linear import PlatonicLinear
from platonic_transformers.models.platoformer_jax.utils import drop_path


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Attributes:
        drop_prob: Probability of dropping a path
        scale_by_keep: Whether to scale output by keep probability
    """
    drop_prob: float = 0.0
    scale_by_keep: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Apply drop path."""
        if self.drop_prob == 0. or deterministic:
            return x
        
        if rng is None:
            rng = self.make_rng('dropout')
        
        return drop_path(x, self.drop_prob, rng, training=not deterministic, scale_by_keep=self.scale_by_keep)


class PlatonicBlock(nn.Module):
    """
    A group-equivariant Transformer-style block using Platonic symmetries.
    
    Replaces standard self-attention and feed-forward networks with equivariant
    counterparts: PlatonicConv and PlatonicLinear.
    
    Attributes:
        d_model: Total model dimension (G * C_model)
        nhead: Number of base attention heads
        dim_feedforward: FFN hidden dimension
        solid_name: Name of the Platonic solid
        dropout: Dropout rate
        drop_path: Stochastic depth rate
        layer_scale_init_value: Initial value for LayerScale (None to disable)
        spatial_dims: Number of spatial dimensions
        freq_sigma: RoPE frequency sigma
        freq_init: Frequency initialization method
        learned_freqs: Whether frequencies are learnable
        mean_aggregation: Whether to normalize by node count
        attention: Whether to use softmax attention
        use_key: Whether to learn separate key projection
    """
    d_model: int
    nhead: int
    dim_feedforward: int
    solid_name: str
    dropout: float = 0.1
    drop_path: float = 0.0
    layer_scale_init_value: Optional[float] = None
    spatial_dims: int = 3
    freq_sigma: float = 1.0
    freq_init: str = 'spiral'
    learned_freqs: bool = True
    mean_aggregation: bool = False
    attention: bool = False
    use_key: bool = False
    
    def setup(self):
        """Initialize the block."""
        self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        self.num_G = self.group.G
        
        # Validate dimensions
        if self.d_model % self.num_G != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by group size ({self.num_G}).")
        if self.dim_feedforward % self.num_G != 0:
            raise ValueError(f"dim_feedforward ({self.dim_feedforward}) must be divisible by group size ({self.num_G}).")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead}).")
        
        self.dim_per_g = self.d_model // self.num_G
        
        # Interaction layer
        self.interaction = PlatonicConv(
            in_channels=self.d_model,
            out_channels=self.d_model,
            embed_dim=self.d_model,
            num_heads=self.nhead,
            solid_name=self.solid_name,
            spatial_dims=self.spatial_dims,
            freq_sigma=self.freq_sigma,
            freq_init=self.freq_init,
            learned_freqs=self.learned_freqs,
            mean_aggregation=self.mean_aggregation,
            attention=self.attention,
            use_key=self.use_key,
        )
        
        # FFN layers
        self.linear1 = PlatonicLinear(self.d_model, self.dim_feedforward, self.solid_name)
        self.linear2 = PlatonicLinear(self.dim_feedforward, self.d_model, self.solid_name)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.norm2 = nn.LayerNorm(epsilon=1e-5)
        
        # Dropout
        self.dropout1 = nn.Dropout(rate=self.dropout)
        self.dropout2 = nn.Dropout(rate=self.dropout)
        self.ffn_dropout = nn.Dropout(rate=self.dropout)
        
        # Drop path
        self.drop_path1 = DropPath(drop_prob=self.drop_path)
        self.drop_path2 = DropPath(drop_prob=self.drop_path)
    
    @nn.compact
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
        Forward pass.
        
        Args:
            x: Input features of shape [..., G*C]
            pos: Positions of shape [..., spatial_dims]
            batch: Batch indices (graph mode)
            mask: Attention mask (dense mode)
            avg_num_nodes: Average number of nodes
            deterministic: Whether in inference mode
            
        Returns:
            Output features of same shape as input
        """
        # Initialize LayerScale if needed
        if self.layer_scale_init_value is not None:
            gamma_1 = self.param('gamma_1', 
                                 nn.initializers.constant(self.layer_scale_init_value),
                                 (self.d_model,))
            gamma_2 = self.param('gamma_2',
                                 nn.initializers.constant(self.layer_scale_init_value),
                                 (self.d_model,))
        else:
            gamma_1 = None
            gamma_2 = None
        
        # Interaction Block (pre-normalization)
        normed_x = self._normalize(x, self.norm1)
        interaction_out = self._interaction_block(normed_x, pos, batch, mask, avg_num_nodes, deterministic)
        
        if gamma_1 is not None:
            interaction_out = gamma_1 * interaction_out
        
        residual = self.drop_path1(interaction_out, deterministic=deterministic)
        x = x + residual
        
        # FFN Block (pre-normalization)
        normed_ff = self._normalize(x, self.norm2)
        ff_output = self._ff_block(normed_ff, deterministic)
        
        if gamma_2 is not None:
            ff_output = gamma_2 * ff_output
        
        residual = self.drop_path2(ff_output, deterministic=deterministic)
        x = x + residual
        
        return x
    
    def _normalize(self, x: jnp.ndarray, norm_layer: nn.LayerNorm) -> jnp.ndarray:
        """Apply LayerNorm on the per-group-element dimension."""
        leading_dims = x.shape[:-1]
        # Reshape: [..., G*C] -> [..., G, C]
        x_reshaped = x.reshape(*leading_dims, self.num_G, self.dim_per_g)
        # Apply normalization
        normed_reshaped = norm_layer(x_reshaped)
        # Reshape back
        return normed_reshaped.reshape(*leading_dims, -1)
    
    def _interaction_block(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        batch: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray],
        avg_num_nodes: float,
        deterministic: bool
    ) -> jnp.ndarray:
        """Wrapper for PlatonicConv layer."""
        interaction_output = self.interaction(x, pos, batch=batch, mask=mask, 
                                             avg_num_nodes=avg_num_nodes, deterministic=deterministic)
        return self.dropout1(interaction_output, deterministic=deterministic)
    
    def _ff_block(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Equivariant Feed-Forward Network block."""
        ff_hidden = self.linear1(x)
        ff_hidden = jax.nn.gelu(ff_hidden)
        ff_hidden = self.ffn_dropout(ff_hidden, deterministic=deterministic)
        ff_output = self.linear2(ff_hidden)
        return self.dropout2(ff_output, deterministic=deterministic)
