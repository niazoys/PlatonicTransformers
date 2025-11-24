import torch
import torch.nn as nn
import math
from torch import Tensor
# This assumes the PLATONIC_GROUPS dictionary from the previous problem is available.
# You might need to adjust the import path based on your project structure.
from .groups import PLATONIC_GROUPS


class PlatonicRoPE(nn.Module):
    """
    Group-Equivariant Rotary Position Embedding (RoPE).

    This module extends Rotary Position Embeddings to be equivariant to the discrete
    rotational symmetry groups of the Platonic solids (T, O, I). It operates on
    feature tensors where the head and group dimensions have been merged for seamless
    integration into standard Multi-Head Attention blocks.

    The core principle is to apply the group action to the spatial coordinates `pos`
    before computing the rotary embeddings. For an input with `H` base heads and a
    group of size `G`, this module effectively has `G*H` heads, where each base
    head's features are rotated according to a different group element.

    Args:
        embed_dim (int): The total embedding dimension, must be divisible by num_heads * num_G * 2.
        num_heads (int): The number of base attention heads.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        spatial_dims (int): The number of spatial dimensions for positions (e.g., 3 for x, y, z).
        freq_sigma (float): Standard deviation for sampling initial random frequencies.
        learned_freqs (bool): If True, frequencies are learnable parameters.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        head_dim: int,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        learned_freqs: bool = False,
        freq_init: str = 'spiral',
    ):
        super().__init__()

        # --- Group Setup ---
        try:
            self.group = PLATONIC_GROUPS[solid_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown solid '{solid_name}'. Available options are {list(PLATONIC_GROUPS.keys())}")
        self.num_G = self.group.G
        self.register_buffer('group_elements', self.group.elements.to(torch.float32))

        # --- Dimension Setup ---
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size ({self.num_G}).")
        self.embed_dim_g = self.embed_dim // self.num_G
        if self.embed_dim_g % self.num_heads != 0:
            raise ValueError(f"embed_dim_g ({self.embed_dim_g}) must be divisible by num_heads ({self.num_heads}).")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 2 for RoPE.")
        self.num_pairs = self.head_dim // 2
        self.spatial_dims = spatial_dims
        self.freq_init = freq_init
        self.freq_sigma = freq_sigma

        # --- Frequency Initialization ---
        # Frequencies are defined per *base* head. The group action is applied to positions.
        # freqs = torch.randn(self.num_heads, self.num_pairs, self.spatial_dims) * freq_sigma
        if self.freq_init == 'random':
            freqs = self._create_random_frequencies()
        elif self.freq_init == 'spiral':
            freqs = self._create_spiral_frequencies()
        else:
            raise ValueError(f"Unknown frequency initialization method: '{self.freq_init}'")


        if learned_freqs:
            self.register_parameter("freqs", nn.Parameter(freqs))
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Apply group-equivariant rotary embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., G, H, D_h). Typically queries or keys.
                        The G*H dimension represents the merged group and head axes.
            pos (Tensor): Position tensor of shape (..., spatial_dims). The leading
                          dimensions '...' must be broadcastable to the input tensor x.

        Returns:
            Tensor: The rotated input tensor `x_rotated` of the same shape (..., G, H, D_h).
        """
        # 1. --- Unpack and Validate Shapes ---
        *leading_dims, G, H, D_h = x.shape
        if G != self.num_G or H != self.num_heads or D_h != self.head_dim:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (..., {self.num_G}, {self.num_heads}, {self.head_dim}).")
        
        # 2. --- Compute Rotated frequencies ---
        freqs_rotated = torch.einsum('gde, hfe -> ghfd', self.group_elements, self.freqs)

        # Compute rotation angles for each rotated position and each base head.
        angles = torch.einsum('...d, ghfd -> ...ghf', pos, freqs_rotated)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 3. --- Apply Rotations to Input Features ---
        # Reshape input features to expose pairs for 2D rotation.
        # Shape: [..., G, H, F, 2]
        x_reshaped = x.view(*leading_dims, self.num_G, self.num_heads, self.num_pairs, 2)
        x0, x1 = x_reshaped.unbind(dim=-1)  # Both have shape [..., G, H, F]

        # Apply the 2D rotation to each pair.
        # The cos/sin angles broadcast across the leading dimensions.
        x_rotated_0 = x0 * cos_angles - x1 * sin_angles
        x_rotated_1 = x0 * sin_angles + x1 * cos_angles
        
        # Stack the rotated pairs back together.
        # Shape: [..., G, H, F, 2]
        x_rotated_pairs = torch.stack([x_rotated_0, x_rotated_1], dim=-1)

        # 4. --- Reshape to Final Output ---
        # Reshape back to the merged (G*H, D_h) convention.
        # Final shape: (..., G, H, D_h)
        x_out = x_rotated_pairs.view(*leading_dims, self.num_G, self.num_heads, self.head_dim)
        
        return x_out

    def _create_random_frequencies(self) -> Tensor:
        return torch.randn(self.num_heads, self.num_pairs, self.spatial_dims) * self.freq_sigma
    
    def _create_spiral_frequencies(self) -> Tensor:
        if self.spatial_dims == 2:
            return self._create_spiral_frequencies_2d()
        elif self.spatial_dims == 3:
            return self._create_spiral_frequencies_3d()
        else:
            raise ValueError("Spiral method currently only supports spatial_dims=2 or 3")
    
    def _create_spiral_frequencies_3d(self) -> Tensor:
        if self.spatial_dims != 3:
            raise ValueError("Spiral method currently only supports spatial_dims=3")

        # 1. Define base indices and magnitudes for the pairs (F dimension)
        indices = torch.arange(0, self.num_pairs, dtype=torch.float32) + 0.5
        magnitudes = torch.linspace(
            self.freq_sigma / self.num_pairs, self.freq_sigma, self.num_pairs
        )
        
        # 2. Create deterministic phase offsets for each head (H dimension)
        # Shape: [num_heads, 1] for broadcasting
        head_phases = torch.linspace(0, 2 * math.pi, self.num_heads + 1)[:-1].unsqueeze(1)

        # 3. Calculate spiral coordinates using broadcasting
        phi = (1 + math.sqrt(5)) / 2
        
        # y and radius are the same for all heads, but need to be broadcastable
        # Shape: [1, num_pairs]
        y = (1 - 2 * indices / self.num_pairs).unsqueeze(0)
        radius = torch.sqrt(1 - y**2)

        # Theta now includes the per-head phase offset
        # base_theta [1, num_pairs] + head_phases [num_heads, 1] -> theta [num_heads, num_pairs]
        base_theta = (2 * math.pi * indices / phi).unsqueeze(0)
        theta = base_theta + head_phases
        
        # Calculate x and z for each head's spiral
        # Shape: [num_heads, num_pairs]
        x = radius * torch.cos(theta)
        z = radius * torch.sin(theta)
        
        # Expand y to match the head dimension
        # Shape: [num_heads, num_pairs]
        y_expanded = y.expand(self.num_heads, -1)
        
        # 4. Stack and combine with magnitudes
        # directions shape: [num_heads, num_pairs, 3]
        directions = torch.stack([x, y_expanded, z], dim=-1)
        
        # magnitudes shape: [1, num_pairs, 1] for broadcasting
        final_freqs = directions * magnitudes.view(1, -1, 1)

        # Final shape is exactly what we need: (num_heads, num_pairs, spatial_dims)
        return final_freqs

    def _create_spiral_frequencies_2d(self) -> Tensor:
        """Generates 2D frequency vectors using a golden angle spiral."""
        indices = torch.arange(0, self.num_pairs, dtype=torch.float32)
        
        # Per-head phase offsets
        head_phases = torch.linspace(0, 2 * math.pi, self.num_heads + 1)[:-1].unsqueeze(1)
        
        # Golden angle for uniform angular distribution
        golden_angle = math.pi * (3. - math.sqrt(5.))
        
        # Base theta and radius
        # Radius scales with sqrt(index) for uniform area coverage
        base_theta = (indices * golden_angle).unsqueeze(0)
        normalized_indices = (indices + 1) / self.num_pairs # Normalize to 1, zero freq not included
        radius = torch.sqrt(normalized_indices).unsqueeze(0) * self.freq_sigma
                                                
        # Add head phases for per-head variation
        theta = base_theta + head_phases
        
        # Convert from polar to Cartesian coordinates
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        
        # Stack and scale by max_freq
        # Shape: [num_heads, num_pairs, 2]
        freq_vectors = torch.stack([x, y], dim=-1)
        
        return freq_vectors
