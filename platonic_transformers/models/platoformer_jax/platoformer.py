"""
Platonic Transformer model for JAX/Flax.

A Transformer architecture equivariant to the symmetries of a specified Platonic solid.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer_jax.block import PlatonicBlock
from platonic_transformers.models.platoformer_jax.linear import PlatonicLinear
from platonic_transformers.models.platoformer_jax.ape import PlatonicAPE
from platonic_transformers.models.platoformer_jax.io import lift, pool, to_scalars_vectors, to_dense_and_mask


class PlatonicTransformer(nn.Module):
    """
    A Transformer architecture equivariant to the symmetries of a specified Platonic solid.
    
    This model processes point cloud data. It first embeds input node features, then
    "lifts" them into a group-equivariant feature space. A series of PlatonicBlocks
    process these features equivariantly. Finally, for graph-level tasks, it pools
    over the nodes and the group to produce a single invariant prediction.
    
    Attributes:
        input_dim: Dimensionality of initial node features
        input_dim_vec: Dimensionality of vector features (0 if none)
        hidden_dim: Per-group-element channel dimension
        output_dim: Dimensionality of scalar output
        output_dim_vec: Dimensionality of vector output (0 if none)
        nhead: Number of attention heads per block
        num_layers: Number of PlatonicBlock layers
        solid_name: Name of the Platonic solid defining the symmetry group
        spatial_dim: Number of spatial dimensions
        dense_mode: Force dense mode even if batch is provided
        scalar_task_level: "node" or "graph" pooling for scalars
        vector_task_level: "node" or "graph" pooling for vectors
        ffn_readout: Whether to use FFN for readout
        mean_aggregation: Whether to use mean pooling
        dropout: Dropout rate
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Initial value for LayerScale
        attention: Whether to use softmax attention
        ffn_dim_factor: FFN hidden dim multiplier
        rope_sigma: RoPE frequency sigma
        ape_sigma: APE frequency sigma (None to disable)
        learned_freqs: Whether frequencies are learnable
        freq_init: Frequency initialization method
        use_key: Whether to learn separate key projection
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
    drop_path_rate: float = 0.0
    layer_scale_init_value: Optional[float] = None
    attention: bool = False
    ffn_dim_factor: int = 4
    rope_sigma: float = 1.0
    ape_sigma: Optional[float] = None
    learned_freqs: bool = True
    freq_init: str = 'random'
    use_key: bool = False
    
    def setup(self):
        """Initialize the model."""
        if self.scalar_task_level not in ["node", "graph"]:
            raise ValueError("scalar_task_level must be 'node' or 'graph'.")
        if self.vector_task_level not in ["node", "graph"]:
            raise ValueError("vector_task_level must be 'node' or 'graph'.")
        
        self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        self.num_G = self.group.G
        
        # Input embedding dimension
        input_total = (self.input_dim + self.input_dim_vec * self.spatial_dim) * self.num_G
        
        # Input embedder
        self.x_embedder = PlatonicLinear(input_total, self.hidden_dim, self.solid_name, use_bias=False)
        
        # Position embedding (optional)
        if self.ape_sigma is not None:
            self.ape = PlatonicAPE(
                embed_dim=self.hidden_dim,
                solid_name=self.solid_name,
                freq_sigma=self.ape_sigma,
                spatial_dims=self.spatial_dim,
                learned_freqs=self.learned_freqs
            )
            self._use_ape = True
        else:
            self._use_ape = False
        
        # Encoder layers
        dim_feedforward = int(self.hidden_dim * self.ffn_dim_factor)
        
        self.layers = [
            PlatonicBlock(
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                solid_name=self.solid_name,
                dropout=self.dropout,
                drop_path=self.drop_path_rate,
                layer_scale_init_value=self.layer_scale_init_value,
                freq_sigma=self.rope_sigma,
                freq_init=self.freq_init,
                learned_freqs=self.learned_freqs,
                spatial_dims=self.spatial_dim,
                mean_aggregation=self.mean_aggregation,
                attention=self.attention,
                use_key=self.use_key,
            )
            for _ in range(self.num_layers)
        ]
        
        # Readout layers
        if self.ffn_readout:
            self.scalar_readout = [
                PlatonicLinear(self.hidden_dim, self.hidden_dim, self.solid_name),
                PlatonicLinear(self.hidden_dim, self.num_G * self.output_dim, self.solid_name)
            ]
            self.vector_readout = [
                PlatonicLinear(self.hidden_dim, self.hidden_dim, self.solid_name),
                PlatonicLinear(self.hidden_dim, self.hidden_dim, self.solid_name),
                PlatonicLinear(self.hidden_dim, self.num_G * self.output_dim_vec * self.spatial_dim, self.solid_name)
            ]
        else:
            self.scalar_readout = [
                PlatonicLinear(self.hidden_dim, self.num_G * self.output_dim, self.solid_name)
            ]
            self.vector_readout = [
                PlatonicLinear(self.hidden_dim, self.num_G * self.output_dim_vec * self.spatial_dim, self.solid_name)
            ]
    
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        batch: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        vec: Optional[jnp.ndarray] = None,
        avg_num_nodes: float = 1.0,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for the Platonic Transformer.
        
        Args:
            x: Input node features of shape (N, input_dim) or (B, N, input_dim)
            pos: Node positions of shape (N, spatial_dim) or (B, N, spatial_dim)
            batch: Batch index for each node of shape (N,) (graph mode only)
            mask: Attention mask of shape (B, N) (dense mode only)
            vec: Vector features of shape (N, input_dim_vec, spatial_dim) or (B, N, input_dim_vec, spatial_dim)
            avg_num_nodes: Average number of nodes for normalization
            deterministic: Whether in inference mode
            
        Returns:
            Tuple of (scalars, vectors):
                - scalars: Scalar predictions of shape (B, output_dim) for graph tasks
                           or (N, output_dim) for node tasks
                - vectors: Vector predictions of shape (B, output_dim_vec, spatial_dim) for graph tasks
                           or (N, output_dim_vec, spatial_dim) for node tasks
        """
        # Track if input was dense
        input_was_dense = (batch is None)
        
        # Convert to dense if needed
        if self.dense_mode:
            x, vec, pos, mask = to_dense_and_mask(x, vec, pos, batch)
            batch = None
        else:
            mask = None
        
        # Lift scalars and vectors, then embed
        x_lifted = lift(x, vec, self.group)
        x_embedded = self.x_embedder(x_lifted)
        
        # Add absolute position embedding
        if self._use_ape:
            x_embedded = x_embedded + self.ape(pos)
        
        # Encoder layers
        x_out = x_embedded
        for layer in self.layers:
            x_out = layer(
                x=x_out,
                pos=pos,
                batch=batch,
                mask=mask,
                avg_num_nodes=avg_num_nodes,
                deterministic=deterministic
            )
        
        # Pooling and readout for scalars
        if self.scalar_task_level == "graph":
            scalar_x = pool(x_out, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            if not input_was_dense and self.dense_mode and mask is not None:
                scalar_x = x_out[mask]
            else:
                scalar_x = x_out
        
        # Pooling and readout for vectors
        if self.vector_task_level == "graph":
            vector_x = pool(x_out, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            if not input_was_dense and self.dense_mode and mask is not None:
                vector_x = x_out[mask]
            else:
                vector_x = x_out
        
        # Apply scalar readout
        for i, layer in enumerate(self.scalar_readout):
            scalar_x = layer(scalar_x)
            if i < len(self.scalar_readout) - 1:
                scalar_x = jax.nn.gelu(scalar_x)
        
        # Apply vector readout
        for i, layer in enumerate(self.vector_readout):
            vector_x = layer(vector_x)
            if i < len(self.vector_readout) - 1:
                vector_x = jax.nn.gelu(vector_x)
        
        # Extract scalar and vector parts
        scalars = to_scalars_vectors(scalar_x, self.output_dim, 0, self.group)[0]
        vectors = to_scalars_vectors(vector_x, 0, self.output_dim_vec, self.group)[1]
        
        return scalars, vectors


def create_platoformer(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_heads: int,
    solid_name: str = "octahedron",
    **kwargs
) -> PlatonicTransformer:
    """
    Factory function to create a PlatonicTransformer with sensible defaults.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension  
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        solid_name: Name of the Platonic solid
        **kwargs: Additional arguments passed to PlatonicTransformer
        
    Returns:
        Configured PlatonicTransformer instance
    """
    defaults = {
        'input_dim_vec': 0,
        'output_dim_vec': 0,
        'ffn_readout': True,
        'dropout': 0.0,
        'attention': True,
        'rope_sigma': 4.0,
        'ape_sigma': 0.5,
        'freq_init': 'spiral',
    }
    defaults.update(kwargs)
    
    return PlatonicTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        nhead=num_heads,
        num_layers=num_layers,
        solid_name=solid_name,
        **defaults
    )
