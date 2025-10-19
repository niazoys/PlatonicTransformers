import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
import torch.nn.functional as F

# Assumes these modules are in your project structure
from .conv import PlatonicConv
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PlatonicBlock(nn.Module):
    """
    A group-equivariant Transformer-style block using Platonic symmetries.

    This block replaces standard self-attention and feed-forward networks with
    equivariant counterparts: PlatonicConv and PlatonicLinear. It operates on
    flattened group feature maps of shape [..., G*C] but internally handles the
    group structure correctly, especially for Layer Normalization.

    Args:
        d_model (int): The total model dimension (G * C_model). Must be divisible
                       by group size and (group_size * nhead).
        nhead (int): The number of base attention heads for the interaction layer.
        dim_feedforward (int): The total dimension of the feed-forward network's
                               hidden layer (G * C_ffn). Must be divisible by G.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        dropout (float): Dropout rate.
        activation (Callable): The activation function for the FFN.
        layer_norm_eps (float): Epsilon for LayerNorm.
        spatial_dims (int): The number of spatial dimensions for positions.
        drop_path (float): Stochastic depth rate. Default: 0.0.
        layer_scale_init_value (Optional[float]): Initial value for LayerScale. If None,
                                                  LayerScale is not used. Default: None.
        **kwargs: Additional keyword arguments for the PlatonicConv layer
                  (e.g., freq_sigma, learned_freqs, avg_pool).
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        solid_name: str,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        spatial_dims: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        freq_sigma: float = 1.0,
        freq_init: str = 'random',
        learned_freqs: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
    ) -> None:
        super().__init__()

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.norm_first = norm_first

        # Validate total dimensions against group size and heads
        if d_model % self.num_G != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by group size ({self.num_G}).")
        if dim_feedforward % self.num_G != 0:
            raise ValueError(f"dim_feedforward ({dim_feedforward}) must be divisible by group size ({self.num_G}).")
        if d_model % (nhead) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_head = {nhead}.")
        
        # Calculate per-group-element dimensions
        self.dim_per_g = d_model // self.num_G
    
        # --- Equivariant Sub-Modules ---
        self.interaction = PlatonicConv(
            in_channels=d_model,
            out_channels=d_model,
            embed_dim=d_model,
            num_heads=nhead,
            solid_name=solid_name,
            spatial_dims=spatial_dims,
            freq_sigma=freq_sigma,
            freq_init=freq_init,
            learned_freqs=learned_freqs,
            mean_aggregation=mean_aggregation,
            attention=attention,
            use_key=use_key,
        )

        # Equivariant Feed-Forward Network
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name)

        # Layer Normalization (acts on the per-group-element channel dimension)
        self.norm1 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.activation = activation

        # --- DropPath and LayerScale ---
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), requires_grad=True) if layer_scale_init_value is not None else None
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), requires_grad=True) if layer_scale_init_value is not None else None


    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        avg_num_nodes = 1.0
    ) -> Tensor:
        """
        Args:
            x (Tensor): Input feature tensor of shape [..., G*C].
            pos (Tensor): Position tensor of shape [..., D_spatial].
            batch (Optional[Tensor]): For graph mode. Batch index for each element.
            mask (Optional[Tensor]): For dense mode. Boolean mask.
        Returns:
            Tensor: Output feature tensor of the same shape [..., G*C].
        """
        # Interaction Block (pre-normalization is always used)
        normed_x = self._normalize(x, self.norm1)
        interaction_out = self._interaction_block(normed_x, pos, batch, mask, avg_num_nodes)
        if self.gamma_1 is not None:
            interaction_out = self.gamma_1 * interaction_out
        residual = self.drop_path1(interaction_out)
        x = x + residual

        # Feed-Forward Block (pre-normalization is always used)
        normed_ff = self._normalize(x, self.norm2)
        ff_output = self._ff_block(normed_ff)
        if self.gamma_2 is not None:
            ff_output = self.gamma_2 * ff_output
        residual = self.drop_path2(ff_output)
        x = x + residual
        
        return x

    def _normalize(self, x: Tensor, norm_layer: nn.LayerNorm) -> Tensor:
        """Helper to apply LayerNorm on the per-group-element dimension."""
        leading_dims = x.shape[:-1]
        # Reshape to expose group axis: [..., G*C] -> [..., G, C]
        x_reshaped = x.view(*leading_dims, self.num_G, self.dim_per_g)
        # Apply normalization
        normed_reshaped = norm_layer(x_reshaped)
        # Reshape back to original convention
        return normed_reshaped.view(*leading_dims, -1)

    def _interaction_block(
        self, x: Tensor, pos: Tensor, batch: Optional[Tensor], mask: Optional[Tensor], avg_num_nodes = 1.0
    ) -> Tensor:
        """Wrapper for the PlatonicConv layer."""
        interaction_output = self.interaction(x, pos, batch=batch, mask=mask, avg_num_nodes=avg_num_nodes)
        return self.dropout1(interaction_output)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Equivariant Feed-Forward Network block."""
        ff_output = self.linear2(self.ffn_dropout(self.activation(self.linear1(x))))
        return self.dropout2(ff_output)
