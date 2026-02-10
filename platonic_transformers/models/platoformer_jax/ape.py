"""
Group-Equivariant Absolute Position Encoding (APE) for JAX/Flax.

This module provides both standard APE and group-equivariant PlatonicAPE.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import math
from typing import Optional

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS


class APE(nn.Module):
    """
    Absolute Position Encoding using random sinusoidal features.
    
    Maps spatial coordinates to a high-dimensional embedding vector using
    random sinusoidal basis functions (Random Fourier Features).
    
    Attributes:
        embed_dim: Total dimension of the output embedding (must be even)
        freq_sigma: Standard deviation for sampling random frequencies
        spatial_dims: Number of spatial dimensions of input positions
        learned_freqs: If True, frequencies become learnable parameters
    """
    embed_dim: int
    freq_sigma: float
    spatial_dims: int = 3
    learned_freqs: bool = False
    
    def setup(self):
        """Validate configuration."""
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {self.embed_dim}.")
        self.num_frequencies = self.embed_dim // 2
    
    @nn.compact
    def __call__(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the sinusoidal position embeddings.
        
        Args:
            pos: Position tensor of shape (..., spatial_dims)
            
        Returns:
            Position embedding of shape (..., embed_dim)
        """
        # Initialize frequencies
        key = jax.random.PRNGKey(42)
        freqs_init = jax.random.normal(key, (self.spatial_dims, self.num_frequencies)) * self.freq_sigma
        
        if self.learned_freqs:
            freqs = self.param('freqs', lambda key: freqs_init)
        else:
            freqs = freqs_init
        
        # Project positions onto frequency vectors
        angles = jnp.einsum('...d, df -> ...f', pos, freqs)
        
        # Compute sinusoidal features
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)
        
        # Concatenate: (..., 2*f) = (..., embed_dim)
        embedding = jnp.concatenate([cos_angles, sin_angles], axis=-1)
        
        return embedding


class PlatonicAPE(nn.Module):
    """
    Group-Equivariant Absolute Position Encoding.
    
    Extends APE to be equivariant to the discrete rotational symmetry groups
    of the Platonic solids. Generates G distinct position embeddings by applying
    each group rotation to the base frequencies.
    
    Attributes:
        embed_dim: Total dimension of output embedding (must be divisible by G)
        solid_name: Name of the Platonic solid defining the symmetry group
        freq_sigma: Standard deviation for sampling random frequencies
        spatial_dims: Number of spatial dimensions of input positions
        learned_freqs: If True, base frequencies become learnable parameters
    """
    embed_dim: int
    solid_name: str
    freq_sigma: float
    spatial_dims: int = 3
    learned_freqs: bool = False
    
    def setup(self):
        """Initialize and validate configuration."""
        try:
            self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown solid '{self.solid_name}'. "
                           f"Available: {list(PLATONIC_GROUPS.keys())}")
        
        self.num_G = self.group.G
        
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size G ({self.num_G}).")
        
        self.embed_dim_g = self.embed_dim // self.num_G
        
        if self.embed_dim_g % 2 != 0:
            raise ValueError(f"embed_dim per group element ({self.embed_dim_g}) must be even.")
        
        self.num_frequencies_g = self.embed_dim_g // 2
    
    @nn.compact
    def __call__(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the group-equivariant sinusoidal position embeddings.
        
        Args:
            pos: Position tensor of shape (..., spatial_dims)
            
        Returns:
            Position embedding of shape (..., embed_dim)
        """
        # Initialize base frequencies
        key = jax.random.PRNGKey(42)
        freqs_init = jax.random.normal(key, (self.spatial_dims, self.num_frequencies_g)) * self.freq_sigma
        
        if self.learned_freqs:
            freqs = self.param('freqs', lambda key: freqs_init)
        else:
            freqs = freqs_init
        
        # Get group elements
        group_elements = self.group.get_elements_jax()
        
        # Rotate base frequencies using group elements
        # group_elements: (G, d, d) | freqs: (d, f_g) -> freqs_rotated: (G, d, f_g)
        freqs_rotated = jnp.einsum('gij, jf -> gif', group_elements, freqs)
        
        # Project positions onto rotated frequencies
        # pos: (..., d) | freqs_rotated: (G, d, f_g) -> angles: (..., G, f_g)
        angles = jnp.einsum('...d, gdf -> ...gf', pos, freqs_rotated)
        
        # Compute sinusoidal features
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)
        
        # Concatenate: (..., G, 2*f_g) = (..., G, embed_dim_g)
        embedding_grouped = jnp.concatenate([cos_angles, sin_angles], axis=-1)
        
        # Flatten: (..., G, embed_dim_g) -> (..., embed_dim)
        *leading_dims, _, _ = embedding_grouped.shape
        embedding = embedding_grouped.reshape(*leading_dims, self.embed_dim)
        
        return embedding
