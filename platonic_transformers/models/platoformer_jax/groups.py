"""
Platonic Solid Groups implementation for JAX.

This module defines the symmetry groups of Platonic solids (tetrahedron, octahedron, 
icosahedron) and other common groups (cyclic, dihedral) for use in equivariant 
neural networks.
"""

import jax.numpy as jnp
import numpy as np
import math
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class PlatonicSolidGroup:
    """
    A class to hold pre-computed data for a Platonic solid symmetry group.
    
    Unlike PyTorch, we use static numpy arrays that are converted to JAX arrays
    when needed, since Flax modules prefer static configuration.
    
    Attributes:
        elements: Array of shape [G, D, D] containing the group elements as rotation matrices.
        G: Order of the group (number of elements).
        dim: Dimensionality of the space (typically 3 for 3D rotations).
        cayley_table: Multiplication table of shape [G, G].
        inverse_indices: Array of shape [G] containing inverse element indices.
    """
    elements: np.ndarray
    G: int
    dim: int
    cayley_table: np.ndarray
    inverse_indices: np.ndarray
    
    def get_elements_jax(self) -> jnp.ndarray:
        """Returns group elements as JAX array."""
        return jnp.array(self.elements, dtype=jnp.float32)
    
    def get_cayley_table_jax(self) -> jnp.ndarray:
        """Returns Cayley table as JAX array."""
        return jnp.array(self.cayley_table, dtype=jnp.int32)
    
    def get_inverse_indices_jax(self) -> jnp.ndarray:
        """Returns inverse indices as JAX array."""
        return jnp.array(self.inverse_indices, dtype=jnp.int32)


def _compute_inverse_indices(elements: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """Computes the index of the inverse for each group element."""
    G = elements.shape[0]
    inverse_indices = np.zeros(G, dtype=np.int64)
    # For orthogonal matrices, inverse is the transpose
    inverses_mat = np.transpose(elements, (0, 2, 1))
    
    for i in range(G):
        diffs = np.sum((elements - inverses_mat[i:i+1])**2, axis=(1, 2))
        j = np.argmin(diffs)
        if diffs[j] < atol**2:
            inverse_indices[i] = j
        else:
            raise RuntimeError(f"Could not find inverse for element {i}")
    return inverse_indices


def _compute_cayley_table(elements: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """Computes the Cayley table (multiplication table) for the group."""
    G = elements.shape[0]
    cayley_table = np.zeros((G, G), dtype=np.int64)
    
    for i in range(G):
        for j in range(G):
            composition = elements[i] @ elements[j]
            diffs = np.sum((elements - composition[np.newaxis])**2, axis=(1, 2))
            k = np.argmin(diffs)
            if diffs[k] < atol**2:
                cayley_table[i, j] = k
            else:
                raise RuntimeError(f"Cayley table construction failed for elements {i} and {j}")
    return cayley_table


def create_platonic_group(elements: np.ndarray, solid_name: str) -> PlatonicSolidGroup:
    """Create a PlatonicSolidGroup from group elements."""
    elements = elements.astype(np.float64)
    G = elements.shape[0]
    
    # Determine dimension
    if solid_name in ["trivial", "tetrahedron", "octahedron", "icosahedron", "octahedron_reflections"]:
        dim = 3
    elif solid_name.startswith("flop_"):
        parts = solid_name.split("_")
        dim = int(parts[1][:-1])  # e.g., "flop_3d_1" -> 3
    elif solid_name.startswith("cyclic") or solid_name.startswith("dihedral"):
        dim = 2
    elif solid_name.startswith("trivial_"):
        dim = int(solid_name.split("_")[1])
    else:
        dim = elements.shape[1]
    
    # Validate orthogonality
    dets = np.linalg.det(elements)
    if not np.allclose(np.abs(dets), np.ones_like(dets), atol=1e-5):
        raise ValueError(f"All elements for group '{solid_name}' must be orthogonal.")
    
    inverse_indices = _compute_inverse_indices(elements)
    cayley_table = _compute_cayley_table(elements)
    
    return PlatonicSolidGroup(
        elements=elements.astype(np.float32),
        G=G,
        dim=dim,
        cayley_table=cayley_table,
        inverse_indices=inverse_indices,
    )


def _get_trivial_elements(dim: int = 3) -> np.ndarray:
    """Returns the single element of the trivial group (the identity)."""
    return np.eye(dim, dtype=np.float32)[np.newaxis, ...]


def _generate_reflection_elements(dim: int, axis: int) -> np.ndarray:
    """Generates the 2 elements of a reflection group."""
    identity = np.eye(dim, dtype=np.float32)
    reflection = identity.copy()
    
    if dim == 2:
        if axis == 1:
            reflection[1, 1] = -1
        elif axis == 2:
            reflection[0, 0] = -1
        else:
            raise ValueError("Axis for 2D reflection must be 1 or 2.")
    else:
        if axis < 1 or axis > dim:
            raise ValueError(f"Axis for reflection must be between 1 and {dim}.")
        reflection[axis - 1, axis - 1] = -1
    
    return np.stack([identity, reflection])


def _generate_cyclic_permutation_elements(n: int) -> np.ndarray:
    """Generates the n rotation matrices of the 2D cyclic group C_n."""
    elements = []
    angle_step = 2 * math.pi / n
    
    for i in range(n):
        angle = i * angle_step
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        elements.append(np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32))
    
    return np.stack(elements)


def _generate_dihedral_elements(n: int) -> np.ndarray:
    """Generates the 2n elements of the 2D dihedral group D_n."""
    rotations = _generate_cyclic_permutation_elements(n)
    reflection = np.array([[1, 0], [0, -1]], dtype=np.float32)
    reflections = rotations @ reflection
    return np.concatenate([rotations, reflections], axis=0)


def _get_tetrahedral_elements() -> np.ndarray:
    """Returns the 12 rotation matrices of the Tetrahedral group."""
    return np.array([
        [[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,-1,0],[0,0,-1]],[[-1,0,0],[0,1,0],[0,0,-1]],[[-1,0,0],[0,-1,0],[0,0,1]],
        [[0,0,1],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]],[[0,0,-1],[1,0,0],[0,-1,0]],[[0,-1,0],[0,0,-1],[1,0,0]],
        [[0,0,1],[-1,0,0],[0,-1,0]],[[0,-1,0],[0,0,1],[-1,0,0]],[[0,0,-1],[-1,0,0],[0,1,0]],[[0,1,0],[0,0,-1],[-1,0,0]],
    ], dtype=np.float32)


def _get_octahedral_elements() -> np.ndarray:
    """Returns the 24 rotation matrices of the Octahedral group."""
    base = _get_tetrahedral_elements()
    c = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32) @ \
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    return np.concatenate([base, np.stack([b @ c for b in base])], axis=0)


def _rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generates a rotation matrix using Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R


def _generate_icosahedral_elements() -> np.ndarray:
    """Generates the 60 rotation matrices of the Icosahedral group."""
    T = _get_tetrahedral_elements().astype(np.float64)
    
    phi = (1 + math.sqrt(5)) / 2.0
    c = _rodrigues_rotation(np.array([phi, 1.0, 0.0], dtype=np.float64), 2 * math.pi / 5)
    
    c_powers = [np.eye(3, dtype=np.float64)]
    for _ in range(4):
        c_powers.append(c_powers[-1] @ c)
    
    icosahedral_elements = np.stack([t @ c_pow for t in T for c_pow in c_powers])
    
    # Remove duplicates
    unique_elements = []
    atol = 1e-5
    for g in icosahedral_elements:
        if all(not np.allclose(g, existing_g, atol=atol) for existing_g in unique_elements):
            unique_elements.append(g)
    
    if len(unique_elements) != 60:
        raise RuntimeError(f"Failed to generate Icosahedral group. Expected 60 elements, got {len(unique_elements)}")
    
    return np.stack(unique_elements).astype(np.float32)


def _get_axis_aligned_reflection_elements() -> np.ndarray:
    """Returns the 8 diagonal matrices with +/-1 on the diagonal (C2 x C2 x C2 group)."""
    elements = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                elements.append(np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=np.float32))
    return np.stack(elements)


# Pre-compute and create global group instances
TRIVIAL_GROUP = create_platonic_group(_get_trivial_elements(), "trivial")
TETRAHEDRAL_GROUP = create_platonic_group(_get_tetrahedral_elements(), "tetrahedron")
OCTAHEDRAL_GROUP = create_platonic_group(_get_octahedral_elements(), "octahedron")
ICOSAHEDRAL_GROUP = create_platonic_group(_generate_icosahedral_elements(), "icosahedron")
OCTAHEDRON_REFLECTIONS_GROUP = create_platonic_group(_get_axis_aligned_reflection_elements(), "octahedron_reflections")


# Dictionary to access groups by name
PLATONIC_GROUPS: Dict[str, PlatonicSolidGroup] = {
    "trivial": TRIVIAL_GROUP,
    "tetrahedron": TETRAHEDRAL_GROUP,
    "octahedron": OCTAHEDRAL_GROUP,
    "icosahedron": ICOSAHEDRAL_GROUP,
    "octahedron_reflections": OCTAHEDRON_REFLECTIONS_GROUP,
}

# Add trivial groups for dimensions 2 to 10
for n in range(2, 11):
    PLATONIC_GROUPS[f"trivial_{n}"] = create_platonic_group(_get_trivial_elements(n), f"trivial_{n}")

# Add 2D and 3D reflection groups
for dim, axes in ((2, (1, 2)), (3, (1, 2, 3))):
    for axis in axes:
        name = f"flop_{dim}d_{axis}"
        PLATONIC_GROUPS[name] = create_platonic_group(_generate_reflection_elements(dim, axis), name)

# Add 2D cyclic groups C_n for n = 2 to 20
for n in range(2, 21):
    name = f"cyclic_{n}"
    PLATONIC_GROUPS[name] = create_platonic_group(_generate_cyclic_permutation_elements(n), name)


def get_group(solid_name: str) -> PlatonicSolidGroup:
    """Get a Platonic group by name."""
    if solid_name.lower() not in PLATONIC_GROUPS:
        raise ValueError(f"Unknown solid '{solid_name}'. Available: {list(PLATONIC_GROUPS.keys())}")
    return PLATONIC_GROUPS[solid_name.lower()]
