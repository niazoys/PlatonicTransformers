import jax
import jax.numpy as jnp
from jax import Array
import math
from typing import Dict

class PlatonicSolidGroup:
    """
    A class to hold and pre-compute the necessary data for a Platonic solid
    symmetry group, including its elements, order, inverse indices, and
    Cayley table for group multiplication.
    """
    def __init__(self, group_elements: Array, solid_name: str = "group"):
        """
        Initializes the group with a tensor of its elements.

        Args:
            group_elements (Array): A JAX array of shape [G, D, D] where G is
                                    the order of the group and D is the dimension.
            solid_name (str): The name of the group.
        """
        self.elements: Array = group_elements.astype(jnp.float64)
        self.G: int = self.elements.shape[0]

        if solid_name in ["trivial", "tetrahedron", "octahedron", "icosahedron", "octahedron_reflections"]:
            self.dim = 3
        elif solid_name.startswith("flop_"):
            parts = solid_name.split("_")
            if len(parts) != 3 or not parts[1].endswith("d"):
                raise ValueError(
                    f"Invalid flop group name '{solid_name}'. Expected format 'flop_<dim>d_<axis>'."
                )
            dim_part = parts[1][:-1]
            axis_part = parts[2]
            if not (dim_part.isdigit() and axis_part.isdigit()):
                raise ValueError(
                    f"Invalid flop group name '{solid_name}'. Expected numeric dimension and axis."
                )
            self.dim = int(dim_part)
        elif solid_name.startswith("cyclic") or solid_name.startswith("dihedral"):
            self.dim = 2
        elif solid_name.startswith("trivial_"):
            try:
                self.dim = int(solid_name.split("_")[1])
            except ValueError:
                raise ValueError(f"Invalid solid name '{solid_name}'. Expected format 'trivial_n' where n is an integer.")
        else:
            raise ValueError(f"Unknown solid '{solid_name}'. Available options are: "
                             f"{list(PLATONIC_GROUPS.keys())}")
        
        # Compute determinants
        dets = jnp.linalg.det(self.elements)
        # For O(D) matrices, determinants must be +1 or -1
        if not jnp.allclose(jnp.abs(dets), jnp.ones_like(dets), atol=1e-5):
            raise ValueError(
                f"All elements for group '{solid_name}' must be orthogonal (determinant=+/-1). "
                f"Found elements with incorrect determinants."
            )
        
        # Check for orthogonality: R^T R = I
        for i in range(self.G):
            product = self.elements[i].T @ self.elements[i]
            identity = jnp.eye(self.dim, dtype=jnp.float64)
            if not jnp.allclose(product, identity, atol=1e-5):
                raise ValueError(f"Element {i} of group '{solid_name}' is not an orthogonal matrix.")

        self.atol = 1e-5
        self.inverse_indices: Array = self._compute_inverse_indices()
        self.cayley_table: Array = self._compute_cayley_table()

    def _compute_inverse_indices(self) -> Array:
        """Computes the index of the inverse for each group element."""
        inverse_indices = jnp.zeros(self.G, dtype=jnp.int32)
        # For orthogonal matrices, inverse is the transpose
        inverses_mat = jnp.transpose(self.elements, (0, 2, 1))
        
        inverse_indices_list = []
        for i in range(self.G):
            # Find which element in the group matches the inverse
            diff = self.elements - inverses_mat[i][jnp.newaxis, :, :]
            diffs = jnp.sum(diff**2, axis=(1, 2))
            j = jnp.argmin(diffs)
            if diffs[j] < self.atol**2:
                inverse_indices_list.append(int(j))
            else:
                raise RuntimeError(f"Could not find inverse for element {i}")
        
        return jnp.array(inverse_indices_list, dtype=jnp.int32)

    def _compute_cayley_table(self) -> Array:
        """Computes the Cayley table (multiplication table) for the group."""
        cayley_table_list = []
        
        for i in range(self.G):
            row = []
            for j in range(self.G):
                composition = self.elements[i] @ self.elements[j]
                # Find the index of the resulting element in the group
                diff = self.elements - composition[jnp.newaxis, :, :]
                diffs = jnp.sum(diff**2, axis=(1, 2))
                k = jnp.argmin(diffs)
                if diffs[k] < self.atol**2:
                    row.append(int(k))
                else:
                    raise RuntimeError(f"Cayley table construction failed. Product of elements {i} and {j} not found in group.")
            cayley_table_list.append(row)
        
        return jnp.array(cayley_table_list, dtype=jnp.int32)

def _get_trivial_elements(dim=3) -> Array:
    """Returns the single element of the trivial group (the identity)."""
    return jnp.eye(dim, dtype=jnp.float32)[jnp.newaxis, :, :]

def _generate_reflection_elements(dim: int, axis: int) -> Array:
    """
    Generates the 2 elements of a reflection group in the specified dimension.

    Args:
        dim (int): The dimensionality of the space (e.g., 2 or 3).
        axis (int): Which coordinate axis to reflect across. For 2D reflections,
                    axis=1 reflects across the x-axis (flips y) and axis=2 reflects
                    across the y-axis (flips x). For higher dimensions, the axis
                    indicates the coordinate whose sign is flipped.
    """
    if dim < 1:
        raise ValueError("Dimension for reflection must be a positive integer.")

    identity = jnp.eye(dim, dtype=jnp.float32)
    reflection = identity.copy()

    if dim == 2:
        if axis == 1:
            reflection = reflection.at[1, 1].set(-1)
        elif axis == 2:
            reflection = reflection.at[0, 0].set(-1)
        else:
            raise ValueError("Axis for 2D reflection must be 1 (x-axis) or 2 (y-axis).")
    else:
        if axis < 1 or axis > dim:
            raise ValueError(f"Axis for reflection must be between 1 and {dim} for {dim}D reflections.")
        reflection = reflection.at[axis - 1, axis - 1].set(-1)

    return jnp.stack([identity, reflection])

def _generate_cyclic_permutation_elements(n: int) -> Array:
    """Generates the n rotation matrices of the 2D cyclic group C_n."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of elements 'n' must be a positive integer.")
    
    elements = []
    angle_step = 2 * math.pi / n
    for i in range(n):
        angle = i * angle_step
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # 2D rotation matrix
        elements.append(jnp.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=jnp.float32))
        
    return jnp.stack(elements)

def _generate_dihedral_elements(n: int) -> Array:
    """
    Generates the 2n elements of the 2D dihedral group D_n, which includes
    n rotations and n reflections.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The order 'n' must be a positive integer.")
    
    rotations = _generate_cyclic_permutation_elements(n)
    reflection = jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)
    reflections = rotations @ reflection
    return jnp.concatenate([rotations, reflections], axis=0)

def _get_tetrahedral_elements() -> Array:
    """Returns the 12 rotation matrices of the Tetrahedral group."""
    return jnp.array([
        [[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,-1,0],[0,0,-1]],[[-1,0,0],[0,1,0],[0,0,-1]],[[-1,0,0],[0,-1,0],[0,0,1]],
        [[0,0,1],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]],[[0,0,-1],[1,0,0],[0,-1,0]],[[0,-1,0],[0,0,-1],[1,0,0]],
        [[0,0,1],[-1,0,0],[0,-1,0]],[[0,-1,0],[0,0,1],[-1,0,0]],[[0,0,-1],[-1,0,0],[0,1,0]],[[0,1,0],[0,0,-1],[-1,0,0]],
    ], dtype=jnp.float32)

def _get_octahedral_elements() -> Array:
    """Returns the 24 rotation matrices of the Octahedral group."""
    base = _get_tetrahedral_elements()
    c = jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=jnp.float32) @ \
        jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=jnp.float32)
    elements = jnp.concatenate([base, jnp.stack([b @ c for b in base])], axis=0)
    return elements

def _generate_icosahedral_elements() -> Array:
    """
    Generates the 60 rotation matrices of the Icosahedral group programmatically
    using coset decomposition of its tetrahedral subgroup.
    """
    def _rodrigues_rotation(axis: Array, angle: float) -> Array:
        """Generates a rotation matrix using Rodrigues' rotation formula."""
        axis = axis / jnp.linalg.norm(axis)
        K = jnp.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]], dtype=jnp.float32)
        I = jnp.eye(3, dtype=jnp.float32)
        R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R

    T = _get_tetrahedral_elements()
    phi = (1 + math.sqrt(5)) / 2.0
    c = _rodrigues_rotation(jnp.array([phi, 1.0, 0.0]), 2 * math.pi / 5)
    c_powers = [jnp.eye(3, dtype=jnp.float32)]
    for _ in range(4): 
        c_powers.append(c_powers[-1] @ c)
    
    icosahedral_elements = jnp.stack([t @ c_pow for t in T for c_pow in c_powers])
    
    unique_elements = []
    atol = 1e-5
    for g in icosahedral_elements:
        is_unique = True
        for existing_g in unique_elements:
            if jnp.allclose(g, existing_g, atol=atol):
                is_unique = False
                break
        if is_unique:
            unique_elements.append(g)
    
    if len(unique_elements) != 60:
        raise RuntimeError(f"Failed to generate Icosahedral group. Expected 60 elements, got {len(unique_elements)}")
    
    return jnp.stack(unique_elements)

def _get_axis_aligned_reflection_elements() -> Array:
    """
    Returns the 8 diagonal matrices with +/-1 on the diagonal (C2 x C2 x C2 group).
    """
    elements = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                elements.append(jnp.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=jnp.float32))
    return jnp.stack(elements)

# Create global instances for each group
TRIVIAL_GROUP = PlatonicSolidGroup(_get_trivial_elements(), "trivial") 
TETRAHEDRAL_GROUP = PlatonicSolidGroup(_get_tetrahedral_elements(), "tetrahedron")
OCTAHEDRAL_GROUP = PlatonicSolidGroup(_get_octahedral_elements(), "octahedron")
ICOSAHEDRAL_GROUP = PlatonicSolidGroup(_generate_icosahedral_elements(), "icosahedron")
OCTAHEDRON_REFLECTIONS_GROUP = PlatonicSolidGroup(_get_axis_aligned_reflection_elements(), "octahedron_reflections")

# Dictionary to easily access groups by name
PLATONIC_GROUPS: Dict[str, PlatonicSolidGroup] = {
    "trivial": TRIVIAL_GROUP,
    "tetrahedron": TETRAHEDRAL_GROUP,
    "octahedron": OCTAHEDRAL_GROUP,
    "icosahedron": ICOSAHEDRAL_GROUP,
    "octahedron_reflections": OCTAHEDRON_REFLECTIONS_GROUP, 
}

# Add the trivial groups for dim 2 to 10
trivial_groups = {
    f"trivial_{n}": PlatonicSolidGroup(_get_trivial_elements(n), f"trivial_{n}") 
    for n in range(2, 11)
}
PLATONIC_GROUPS.update(trivial_groups)

# Add the 2D reflection groups (flop_1 and flop_2)
flop_groups = {
    f"flop_{dim}d_{axis}": PlatonicSolidGroup(_generate_reflection_elements(dim, axis), f"flop_{dim}d_{axis}")
    for dim, axes in ((2, (1, 2)), (3, (1, 2, 3)))
    for axis in axes
}
PLATONIC_GROUPS.update(flop_groups)

# Add the 2D cyclic groups C_n for n = 2 to 20
cyclic_groups = {
    f"cyclic_{n}": PlatonicSolidGroup(_generate_cyclic_permutation_elements(n), f"cyclic_{n}") 
    for n in range(2, 21)
}
PLATONIC_GROUPS.update(cyclic_groups)

# Add the 2D dihedral groups D_n for n = 2 to 20
dihedral_groups = {
    f"dihedral_{n}": PlatonicSolidGroup(_generate_dihedral_elements(n), f"dihedral_{n}")
    for n in range(2, 21)
}
PLATONIC_GROUPS.update(dihedral_groups)