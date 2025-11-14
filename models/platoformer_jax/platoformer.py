import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from models.platoformer_jax.block import PlatonicBlock

class PlatonicTransformer(nn.Module):
    """
    A Platonic Transformer architecture written in Jax.

    Args:
        input_dim (int): Dimensionality of the initial node features.
        hidden_dim (int): The per-group-element channel dimension used throughout the model.
        output_dim (int): Dimensionality of the final output.
        nhead (int): Number of attention heads in each PlatonicBlock.
        num_layers (int): Number of PlatonicBlock layers.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        ffn_dim_factor (int): Multiplier for the feed-forward network's hidden dimension,
                              relative to `hidden_dim`.
        scalar_task_level (str): "node" or "graph". Determines the pooling strategy.
        dropout (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        layer_scale_init_value (Optional[float]): Initial value for LayerScale. Default: None.
        **kwargs: Additional keyword arguments for the PlatonicBlock layers
    """
    input_dim: int
    input_dim_vec: int
    hidden_dim: int
    output_dim: int
    output_dim_vec: int
    nhead: int
    num_layers: int
    solid_name: str
    spatial_dim: int = 3
    dense_mode: bool = False
    scalar_task_level: str = "graph"
    vector_task_level: str = "node"
    ffn_readout: bool = True
    mean_aggregation: bool = False
    dropout: float = 0.1
    norm_first: bool = True
    drop_path_rate: float = 0.0
    layer_scale_init_value: Optional[float] = None

    @nn.compact
    def __call__(self, x, pos, batch, mask, vec, avg_num_nodes):
        """
        1. x, vec, pos, mask = to_dense_and_mask(x, vec, pos, batch)
        2. x = lift(x, vec, self.group)
        """
        pass