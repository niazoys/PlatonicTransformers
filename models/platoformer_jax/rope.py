import jax
import jax.numpy as jnp
from jax import Array, random
from flax import linen as nn
import math

from models.platoformer_jax.groups import PLATONIC_GROUPS

class PlatonicRoPE(nn.Module):
    embed_dim: int
    num_heads: int
    solid: str
    head_dim: int
    spatial_dims: int = 3
    freq_sigma: float = 1.0
    learned_freqs: bool = False
    freq_init: str = 'spiral'

    def setup(self):
        if self.solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(f"Solid '{self.solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}")
        
        group = PLATONIC_GROUPS[self.solid.lower()]
        self.G = group.G
        
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size ({self.G}).")
        
        self.embed_dim_g = self.embed_dim // self.G
        
        if self.embed_dim_g % self.num_heads != 0:
            raise ValueError(f"embed_dim_g ({self.embed_dim_g}) must be divisible by num_heads ({self.num_heads}).")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 2 for RoPE.")

        self.num_pairs = self.head_dim // 2

        if self.freq_init == 'random':
            freqs = self._create_random_frequencies()
        elif self.freq_init == 'spiral':
            freqs = self._create_spiral_frequencies()
        else:
            raise ValueError(f"Unknown frequency initialization method: '{self.freq_init}'")
        
        if self.learned_freqs:
            self.freqs = self.param("freqs", lambda rng: freqs)
        else:
            self.freqs = freqs

    def __call__(self, x: Array, pos: Array) -> Array:
        *leading_dims, G, H, D_h = x.shape
        if G != self.G or H != self.num_heads or D_h != self.head_dim:
            raise ValueError(f"Input shape (..., G, H, D_h) = {x.shape} does not match expected (G={self.G}, H={self.num_heads}, D_h={self.head_dim})")
        
        freqs_rotated = jnp.einsum('ged, hfe -> ghfd', self.freqs, pos)