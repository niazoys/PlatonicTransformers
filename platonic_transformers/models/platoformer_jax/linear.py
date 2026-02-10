"""
Group-equivariant linear layers for JAX/Flax.

Implements the PlatonicLinear layer which is constrained to be a group convolution
over a Platonic Solid group.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import math
from typing import Optional

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS, PlatonicSolidGroup


class PlatonicLinear(nn.Module):
    """
    A Linear layer constrained to be a group convolution over a Platonic Solid group.
    
    This is the JAX/Flax equivalent of the PyTorch PlatonicLinear module.
    
    Attributes:
        in_features: Number of input features (must be divisible by group order G)
        out_features: Number of output features (must be divisible by group order G)
        solid_name: Name of the Platonic solid defining the symmetry group
        use_bias: Whether to include a bias term
    """
    in_features: int
    out_features: int
    solid_name: str
    use_bias: bool = True
    
    def setup(self):
        """Initialize the layer."""
        if self.solid_name.lower() not in PLATONIC_GROUPS:
            raise ValueError(f"Solid '{self.solid_name}' not recognized. "
                           f"Available: {list(PLATONIC_GROUPS.keys())}")
        
        self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        self.G = self.group.G
        
        if self.in_features % self.G != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by group order {self.G}")
        if self.out_features % self.G != 0:
            raise ValueError(f"out_features ({self.out_features}) must be divisible by group order {self.G}")
        
        self.in_channels = self.in_features // self.G
        self.out_channels = self.out_features // self.G
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the group-equivariant linear transformation.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Calculate the effective fan-in for proper initialization
        fan_in = self.G * self.in_channels
        std = 1.0 / math.sqrt(fan_in)
        
        # Initialize the kernel: shape [G, out_channels, in_channels]
        kernel = self.param(
            'kernel',
            nn.initializers.normal(std),
            (self.G, self.out_channels, self.in_channels)
        )
        
        if self.use_bias:
            bound = 1.0 / math.sqrt(fan_in)
            bias = self.param(
                'bias',
                nn.initializers.uniform(bound),
                (self.out_channels,)
            )
        else:
            bias = None
        
        # Get group data
        cayley_table = self.group.get_cayley_table_jax()
        inverse_indices = self.group.get_inverse_indices_jax()
        
        # Construct the full weight matrix
        weight = self._get_weight(kernel, cayley_table, inverse_indices)
        
        # Apply linear transformation
        output = jnp.dot(x, weight.T)
        
        # Apply bias (broadcasted over group dimension)
        if bias is not None:
            output_shape = output.shape
            output = output.reshape(*output_shape[:-1], self.G, self.out_channels)
            output = output + bias
            output = output.reshape(output_shape)
        
        return output
    
    def _get_weight(self, 
                    kernel: jnp.ndarray,
                    cayley_table: jnp.ndarray,
                    inverse_indices: jnp.ndarray) -> jnp.ndarray:
        """
        Constructs the full [G*O, G*I] weight matrix from the fundamental kernel.
        
        Args:
            kernel: Fundamental kernel of shape [G, out_channels, in_channels]
            cayley_table: Group multiplication table of shape [G, G]
            inverse_indices: Inverse element indices of shape [G]
            
        Returns:
            Full weight matrix of shape [out_features, in_features]
        """
        G = self.G
        
        # Create index grids
        h_indices = jnp.arange(G).reshape(G, 1)
        g_indices = jnp.arange(G).reshape(1, G)
        
        # Compute kernel indices using group structure
        inv_g_indices = inverse_indices[g_indices]
        kernel_group_idx = cayley_table[inv_g_indices, h_indices]
        
        # Gather kernel values
        expanded_kernel = kernel[kernel_group_idx]
        
        # Reshape to full weight matrix
        weight = expanded_kernel.transpose(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        
        return weight


class EquivariantLayerNorm(nn.Module):
    """
    Layer normalization that respects group equivariance.
    
    Applies layer normalization to the per-group-element channel dimension,
    ensuring equivariance is preserved.
    
    Attributes:
        dim_per_g: Dimension per group element
        num_G: Number of group elements
        epsilon: Small constant for numerical stability
    """
    dim_per_g: int
    num_G: int
    epsilon: float = 1e-5
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply equivariant layer normalization.
        
        Args:
            x: Input tensor of shape [..., G*C]
            
        Returns:
            Normalized tensor of same shape
        """
        leading_dims = x.shape[:-1]
        
        # Reshape to expose group axis: [..., G*C] -> [..., G, C]
        x_reshaped = x.reshape(*leading_dims, self.num_G, self.dim_per_g)
        
        # Apply layer norm on the channel dimension
        norm = nn.LayerNorm(epsilon=self.epsilon)
        normed = norm(x_reshaped)
        
        # Reshape back
        return normed.reshape(*leading_dims, -1)
