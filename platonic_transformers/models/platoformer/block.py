import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
import torch.nn.functional as F

from platonic_transformers.models.platoformer.conv import PlatonicConv
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS

# Use apex FusedLayerNorm when available (fused CUDA kernel, ~2x faster)
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ImportError:
    LayerNorm = nn.LayerNorm

# Use quack's fused RMSNorm kernel when available (H100+)
try:
    from quack import rmsnorm as _quack_rmsnorm
except ImportError:
    _quack_rmsnorm = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if _quack_rmsnorm is not None and x.is_cuda:
            return _quack_rmsnorm(x, self.weight, eps=self.eps)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


def get_norm_layer(norm_type: str = "layernorm"):
    """Return a norm class based on the requested type.

    Args:
        norm_type: One of "layernorm" (default, uses apex FusedLayerNorm if
                   available) or "rmsnorm" (uses quack fused kernel on GPU,
                   pure-PyTorch fallback on CPU).
    """
    if norm_type == "rmsnorm":
        return RMSNorm
    return LayerNorm


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
        conditioning_dim (Optional[int]): If set, enables DiT-style AdaLN modulation.
                                          The block expects a per-graph conditioning tensor
                                          of shape (B, conditioning_dim) in forward().
                                          Shift/scale/gate are shared across the group axis
                                          to preserve equivariance. Default: None.
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
        norm_type: str = "layernorm",
        spatial_dims: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        freq_sigma: float = 1.0,
        freq_init: str = 'random',
        learned_freqs: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
        rope_on_values: bool = False,
        attention_backend: str = "scatter",
        conditioning_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G

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
            rope_on_values=rope_on_values,
            attention_backend=attention_backend,
        )

        # Equivariant Feed-Forward Network
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name)

        # Normalization (acts on the per-group-element channel dimension)
        NormClass = get_norm_layer(norm_type)
        self.norm1 = NormClass(self.dim_per_g, eps=layer_norm_eps)
        self.norm2 = NormClass(self.dim_per_g, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.activation = activation

        # --- DropPath and LayerScale ---
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # LayerScale: per-channel scaling, shared across group axis for equivariance.
        # Shape (C,) not (G*C,) — the group acts by permuting G, so gamma must be
        # constant across G to commute with the group action.
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((self.dim_per_g)), requires_grad=True) if layer_scale_init_value is not None else None
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((self.dim_per_g)), requires_grad=True) if layer_scale_init_value is not None else None

        # --- AdaLN modulation for diffusion-style conditioning ---
        # Output is 6 * dim_per_g (not 6 * d_model): shift/scale/gate are shared
        # across the group axis to commute with the group action, as with LayerScale.
        # Zero-init: at step 0 the block acts as identity (gate=0, scale=0, shift=0).
        self.conditioning_dim = conditioning_dim
        if conditioning_dim is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conditioning_dim, 6 * self.dim_per_g, bias=True),
            )
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        else:
            self.adaLN_modulation = None


    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
        avg_num_nodes = 1.0
    ) -> Tensor:
        """
        Args:
            x (Tensor): Input feature tensor of shape [..., G*C].
            pos (Tensor): Position tensor of shape [..., D_spatial].
            batch (Optional[Tensor]): For graph mode. Batch index for each element.
            mask (Optional[Tensor]): For dense mode. Boolean mask.
            conditioning (Optional[Tensor]): Per-graph conditioning of shape (B, conditioning_dim).
                Only used if this block was built with conditioning_dim set.
        Returns:
            Tensor: Output feature tensor of the same shape [..., G*C].
        """
        # Compute AdaLN modulation parameters (if conditioning is active)
        shift_msa = scale_msa = gate_msa = shift_ffn = scale_ffn = gate_ffn = None
        if self.adaLN_modulation is not None and conditioning is not None:
            cond_params = self.adaLN_modulation(conditioning)  # (B, 6*C)
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = [
                self._broadcast_cond(chunk, x, batch)
                for chunk in cond_params.chunk(6, dim=-1)
            ]

        # Interaction Block (pre-normalization is always used)
        normed_x = self._normalize(x, self.norm1)
        normed_x = self._apply_shift_scale(normed_x, shift_msa, scale_msa)
        interaction_out = self._interaction_block(normed_x, pos, batch, mask, avg_num_nodes)
        if self.gamma_1 is not None:
            interaction_out = self._apply_layer_scale(interaction_out, self.gamma_1)
        residual = self.drop_path1(interaction_out)
        x = x + (gate_msa * residual if gate_msa is not None else residual)

        # Feed-Forward Block (pre-normalization is always used)
        normed_ff = self._normalize(x, self.norm2)
        normed_ff = self._apply_shift_scale(normed_ff, shift_ffn, scale_ffn)
        ff_output = self._ff_block(normed_ff)
        if self.gamma_2 is not None:
            ff_output = self._apply_layer_scale(ff_output, self.gamma_2)
        residual = self.drop_path2(ff_output)
        x = x + (gate_ffn * residual if gate_ffn is not None else residual)

        return x

    def _broadcast_cond(self, param: Tensor, x: Tensor, batch: Optional[Tensor]) -> Tensor:
        """Broadcast a (B, C) conditioning chunk to match features of shape (..., G*C).

        The chunk is shared across the group axis to preserve equivariance: the group
        acts by permuting G, so the modulation must be constant across G to commute
        with the action. Output shape matches x for elementwise combination.
        """
        param = param.to(dtype=x.dtype)  # (B, C)
        # Expand across group axis: (B, C) -> (B, G, C) -> (B, G*C)
        param_gc = param[:, None, :].expand(-1, self.num_G, -1).reshape(param.shape[0], -1)
        if x.dim() == 3:
            # Dense mode: x is (B, N, G*C). Insert node axis: (B, 1, G*C).
            return param_gc.unsqueeze(1)
        if x.dim() == 2:
            # Sparse mode: x is (N, G*C). Index by batch: (N, G*C).
            if batch is None:
                raise ValueError("Batch indices are required for sparse-mode conditioning.")
            return param_gc[batch]
        raise ValueError(f"Unsupported tensor rank {x.dim()} for conditioning broadcast.")

    @staticmethod
    def _apply_shift_scale(x: Tensor, shift: Optional[Tensor], scale: Optional[Tensor]) -> Tensor:
        """Apply AdaLN shift/scale: x * (1 + scale) + shift. No-op if both None."""
        if shift is None or scale is None:
            return x
        return x * (1 + scale) + shift

    def _apply_layer_scale(self, x: Tensor, gamma: Tensor) -> Tensor:
        """Apply LayerScale gamma (C,) to features (..., G*C), shared across G."""
        leading_dims = x.shape[:-1]
        x_reshaped = x.view(*leading_dims, self.num_G, self.dim_per_g)
        scaled = gamma * x_reshaped
        return scaled.view(*leading_dims, -1)

    def _normalize(self, x: Tensor, norm_layer: nn.Module) -> Tensor:
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
