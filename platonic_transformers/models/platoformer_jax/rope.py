"""
Group-Equivariant Rotary Position Embedding (RoPE) for JAX/Flax.

This module extends Rotary Position Embeddings to be equivariant to the discrete
rotational symmetry groups of the Platonic solids.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import math
from typing import Optional

from platonic_transformers.models.platoformer_jax.groups import PLATONIC_GROUPS


class PlatonicRoPE(nn.Module):
    """
    Group-Equivariant Rotary Position Embedding (RoPE).
    
    This module extends Rotary Position Embeddings to be equivariant to the discrete
    rotational symmetry groups of the Platonic solids (T, O, I).
    
    Attributes:
        embed_dim: Total embedding dimension
        num_heads: Number of base attention heads
        head_dim: Dimension per head
        solid_name: Name of the Platonic solid defining the symmetry group
        spatial_dims: Number of spatial dimensions for positions (e.g., 3 for x, y, z)
        freq_sigma: Standard deviation for sampling initial random frequencies
        learned_freqs: If True, frequencies are learnable parameters
        freq_init: Initialization method ('random' or 'spiral')
    """
    embed_dim: int
    num_heads: int
    head_dim: int
    solid_name: str
    spatial_dims: int = 3
    freq_sigma: float = 1.0
    learned_freqs: bool = False
    freq_init: str = 'spiral'
    
    def setup(self):
        """Initialize the module."""
        try:
            self.group = PLATONIC_GROUPS[self.solid_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown solid '{self.solid_name}'. "
                           f"Available: {list(PLATONIC_GROUPS.keys())}")
        
        self.num_G = self.group.G
        
        # Validate dimensions
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size ({self.num_G}).")
        self.embed_dim_g = self.embed_dim // self.num_G
        
        if self.embed_dim_g % self.num_heads != 0:
            raise ValueError(f"embed_dim_g ({self.embed_dim_g}) must be divisible by num_heads ({self.num_heads}).")
        
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 2 for RoPE.")
        
        self.num_pairs = self.head_dim // 2
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply group-equivariant rotary embeddings to the input tensor.
        
        Args:
            x: Input tensor of shape (..., G, H, D_h). Typically queries or keys.
            pos: Position tensor of shape (..., spatial_dims).
            
        Returns:
            The rotated input tensor of the same shape (..., G, H, D_h).
        """
        # Create or get frequencies
        if self.freq_init == 'random':
            freqs_init = self._create_random_frequencies()
        elif self.freq_init == 'spiral':
            freqs_init = self._create_spiral_frequencies()
        else:
            raise ValueError(f"Unknown frequency initialization method: '{self.freq_init}'")
        
        if self.learned_freqs:
            freqs = self.param('freqs', lambda key: freqs_init)
        else:
            freqs = freqs_init
        
        # Get group elements
        group_elements = self.group.get_elements_jax()
        
        # Validate input shape
        *leading_dims, G, H, D_h = x.shape
        if G != self.num_G or H != self.num_heads or D_h != self.head_dim:
            raise ValueError(f"Input shape {x.shape} does not match expected (..., {self.num_G}, {self.num_heads}, {self.head_dim}).")
        
        # Compute rotated frequencies: [G, H, num_pairs, spatial_dims]
        freqs_rotated = jnp.einsum('gde, hfe -> ghfd', group_elements, freqs)
        
        # Compute rotation angles: [..., G, H, num_pairs]
        angles = jnp.einsum('...d, ghfd -> ...ghf', pos, freqs_rotated)
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)
        
        # Reshape input features to expose pairs: [..., G, H, num_pairs, 2]
        x_reshaped = x.reshape(*leading_dims, self.num_G, self.num_heads, self.num_pairs, 2)
        x0 = x_reshaped[..., 0]  # [..., G, H, num_pairs]
        x1 = x_reshaped[..., 1]
        
        # Apply 2D rotation to each pair
        x_rotated_0 = x0 * cos_angles - x1 * sin_angles
        x_rotated_1 = x0 * sin_angles + x1 * cos_angles
        
        # Stack back: [..., G, H, num_pairs, 2]
        x_rotated_pairs = jnp.stack([x_rotated_0, x_rotated_1], axis=-1)
        
        # Reshape to output: [..., G, H, D_h]
        x_out = x_rotated_pairs.reshape(*leading_dims, self.num_G, self.num_heads, self.head_dim)
        
        return x_out
    
    def _create_random_frequencies(self) -> jnp.ndarray:
        """Create random frequency vectors."""
        # Use a fixed key for reproducibility during initialization
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, (self.num_heads, self.num_pairs, self.spatial_dims)) * self.freq_sigma
    
    def _create_spiral_frequencies(self) -> jnp.ndarray:
        """Create spiral frequency vectors for better coverage."""
        if self.spatial_dims == 2:
            return self._create_spiral_frequencies_2d()
        elif self.spatial_dims == 3:
            return self._create_spiral_frequencies_3d()
        else:
            raise ValueError("Spiral method currently only supports spatial_dims=2 or 3")
    
    def _create_spiral_frequencies_3d(self) -> jnp.ndarray:
        """Generate 3D frequency vectors using a Fibonacci spiral on a sphere."""
        # Define base indices and magnitudes
        indices = jnp.arange(0, self.num_pairs, dtype=jnp.float32) + 0.5
        magnitudes = jnp.linspace(
            self.freq_sigma / self.num_pairs, self.freq_sigma, self.num_pairs
        )
        
        # Create phase offsets for each head
        head_phases = jnp.linspace(0, 2 * math.pi, self.num_heads + 1)[:-1].reshape(-1, 1)
        
        # Calculate spiral coordinates
        phi = (1 + math.sqrt(5)) / 2
        
        # y and radius
        y = (1 - 2 * indices / self.num_pairs).reshape(1, -1)
        radius = jnp.sqrt(1 - y**2)
        
        # Theta with per-head phase offset
        base_theta = (2 * math.pi * indices / phi).reshape(1, -1)
        theta = base_theta + head_phases
        
        # Calculate x and z
        x = radius * jnp.cos(theta)
        z = radius * jnp.sin(theta)
        y_expanded = jnp.broadcast_to(y, (self.num_heads, self.num_pairs))
        
        # Stack and combine with magnitudes
        directions = jnp.stack([x, y_expanded, z], axis=-1)
        final_freqs = directions * magnitudes.reshape(1, -1, 1)
        
        return final_freqs
    
    def _create_spiral_frequencies_2d(self) -> jnp.ndarray:
        """Generate 2D frequency vectors using a golden angle spiral."""
        indices = jnp.arange(0, self.num_pairs, dtype=jnp.float32)
        
        # Per-head phase offsets
        head_phases = jnp.linspace(0, 2 * math.pi, self.num_heads + 1)[:-1].reshape(-1, 1)
        
        # Golden angle
        golden_angle = math.pi * (3. - math.sqrt(5.))
        
        # Base theta and radius
        base_theta = (indices * golden_angle).reshape(1, -1)
        normalized_indices = (indices + 1) / self.num_pairs
        radius = jnp.sqrt(normalized_indices).reshape(1, -1) * self.freq_sigma
        
        # Add head phases
        theta = base_theta + head_phases
        
        # Convert to Cartesian
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        
        # Stack: [num_heads, num_pairs, 2]
        freq_vectors = jnp.stack([x, y], axis=-1)
        
        return freq_vectors
