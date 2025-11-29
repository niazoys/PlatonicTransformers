import torch
from torch import Tensor
import math
from typing import Dict

class PlatonicSolidGroup:
    """
    A class to hold and pre-compute the necessary data for a Platonic solid
    symmetry group, including its elements, order, inverse indices, and
    Cayley table for group multiplication.
    """
    def __init__(self, group_elements: Tensor, solid_name: str = "group"):
        """
        Initializes the group with a tensor of its elements.

        Args:
            group_elements (Tensor): A tensor of shape [G, D, D] where G is
                                     the order of the group and D is the dimension.
            solid_name (str): The name of the group.
        """
        self.elements: Tensor = group_elements.to(dtype=torch.float64)
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
        
        try:
            dets = torch.linalg.det(self.elements)
            # For O(D) matrices, determinants must be +1 or -1
            if not torch.allclose(torch.abs(dets), torch.ones_like(dets), atol=1e-5):
                raise ValueError(
                    f"All elements for group '{solid_name}' must be orthogonal (determinant=+/-1). "
                    f"Found elements with incorrect determinants."
                )
            
            # Check for orthogonality: R^T R = I
            for i in range(self.G):
                if not torch.allclose(self.elements[i].T @ self.elements[i], torch.eye(self.dim, dtype=torch.float64), atol=1e-5):
                    raise ValueError(f"Element {i} of group '{solid_name}' is not an orthogonal matrix.")

        except torch.linalg.LinAlgError as e:
            raise ValueError(f"Could not compute properties for group '{solid_name}'. Error: {e}")

        self.atol = 1e-5
        self.inverse_indices: Tensor = self._compute_inverse_indices()
        self.cayley_table: Tensor = self._compute_cayley_table()

    def _compute_inverse_indices(self) -> Tensor:
        """Computes the index of the inverse for each group element."""
        inverse_indices = torch.zeros(self.G, dtype=torch.long)
        # For orthogonal matrices, inverse is the transpose
        inverses_mat = self.elements.transpose(-1, -2)
        for i in range(self.G):
            # Find which element in the group matches the inverse
            diffs = torch.sum((self.elements - inverses_mat[i].unsqueeze(0))**2, dim=(1, 2))
            j = torch.argmin(diffs)
            if diffs[j] < self.atol**2:
                inverse_indices[i] = j
            else:
                raise RuntimeError(f"Could not find inverse for element {i}")
        return inverse_indices

    def _compute_cayley_table(self) -> Tensor:
        """Computes the Cayley table (multiplication table) for the group."""
        cayley_table = torch.zeros((self.G, self.G), dtype=torch.long)
        for i in range(self.G):
            for j in range(self.G):
                composition = self.elements[i] @ self.elements[j]
                # Find the index of the resulting element in the group
                diffs = torch.sum((self.elements - composition.unsqueeze(0))**2, dim=(1, 2))
                k = torch.argmin(diffs)
                if diffs[k] < self.atol**2:
                    cayley_table[i, j] = k
                else:
                    raise RuntimeError(f"Cayley table construction failed. Product of elements {i} and {j} not found in group.")
        return cayley_table

def _get_trivial_elements(dim=3) -> Tensor:
    """Returns the single element of the trivial group (the identity)."""
    return torch.eye(dim, dtype=torch.float32).unsqueeze(0)

def _generate_reflection_elements(dim: int, axis: int) -> Tensor:
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

    identity = torch.eye(dim, dtype=torch.float32)
    reflection = identity.clone()

    if dim == 2:
        if axis == 1:
            reflection[1, 1] = -1
        elif axis == 2:
            reflection[0, 0] = -1
        else:
            raise ValueError("Axis for 2D reflection must be 1 (x-axis) or 2 (y-axis).")
    else:
        if axis < 1 or axis > dim:
            raise ValueError(f"Axis for reflection must be between 1 and {dim} for {dim}D reflections.")
        reflection[axis - 1, axis - 1] = -1

    return torch.stack([identity, reflection])

def _generate_cyclic_permutation_elements(n: int) -> Tensor:
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
        elements.append(torch.tensor([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=torch.float32))
        
    return torch.stack(elements)

def _generate_dihedral_elements(n: int) -> Tensor:
    """
    Generates the 2n elements of the 2D dihedral group D_n, which includes
    n rotations and n reflections.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The order 'n' must be a positive integer.")
    
    rotations = _generate_cyclic_permutation_elements(n)
    reflection = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
    reflections = rotations @ reflection
    return torch.cat([rotations, reflections], dim=0)

def _get_tetrahedral_elements(dtype=torch.float32) -> Tensor:
    """Returns the 12 rotation matrices of the Tetrahedral group."""
    return torch.tensor([
        [[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,-1,0],[0,0,-1]],[[-1,0,0],[0,1,0],[0,0,-1]],[[-1,0,0],[0,-1,0],[0,0,1]],
        [[0,0,1],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]],[[0,0,-1],[1,0,0],[0,-1,0]],[[0,-1,0],[0,0,-1],[1,0,0]],
        [[0,0,1],[-1,0,0],[0,-1,0]],[[0,-1,0],[0,0,1],[-1,0,0]],[[0,0,-1],[-1,0,0],[0,1,0]],[[0,1,0],[0,0,-1],[-1,0,0]],
    ], dtype=dtype)

def _get_octahedral_elements() -> Tensor:
    """Returns the 24 rotation matrices of the Octahedral group."""
    base = _get_tetrahedral_elements()
    c = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32) @ \
        torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    elements = torch.cat([base, torch.stack([b @ c for b in base])], dim=0)
    return elements

def _generate_icosahedral_elements() -> Tensor:
    """
    Generates the 60 rotation matrices of the Icosahedral group programmatically
    using coset decomposition of its tetrahedral subgroup.
    """
    # FORCE FLOAT64 for generation to avoid error accumulation
    gen_dtype = torch.float64

    def _rodrigues_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
        """Generates a rotation matrix using Rodrigues' rotation formula."""
        axis = axis / torch.linalg.norm(axis)
        # Explicitly use gen_dtype for intermediate tensors
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=gen_dtype)
        I = torch.eye(3, dtype=gen_dtype)
        R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R

    # Get base elements in high precision
    T = _get_tetrahedral_elements(dtype=gen_dtype)
    
    phi = (1 + math.sqrt(5)) / 2.0
    c = _rodrigues_rotation(torch.tensor([phi, 1.0, 0.0], dtype=gen_dtype), 2 * math.pi / 5)
    
    c_powers = [torch.eye(3, dtype=gen_dtype)]
    for _ in range(4): c_powers.append(c_powers[-1] @ c)
    
    icosahedral_elements = torch.stack([t @ c_pow for t in T for c_pow in c_powers])
    
    unique_elements = []
    atol = 1e-5
    for g in icosahedral_elements:
        if all(not torch.allclose(g, existing_g, atol=atol) for existing_g in unique_elements):
            unique_elements.append(g)
            
    if len(unique_elements) != 60:
        raise RuntimeError(f"Failed to generate Icosahedral group. Expected 60 elements, got {len(unique_elements)}")
        
    # Convert back to standard float32 for storage (optional, usually handled by class)
    return torch.stack(unique_elements).to(torch.float32)

def _get_axis_aligned_reflection_elements() -> Tensor:
    """
    Returns the 8 diagonal matrices with +/-1 on the diagonal (C2 x C2 x C2 group).
    """
    elements = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                elements.append(torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=torch.float32))
    return torch.stack(elements)

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

# # Add 3D reflection group about the different axes
# PLATONIC_GROUPS.update({
#     "flip_3d_axis0": PlatonicSolidGroup(
#         torch.stack([
#             torch.eye(3, dtype=torch.float32),
#             torch.diag(torch.tensor([-1, 1, 1], dtype=torch.float32)),
#         ]),
#         "flip_3d_axis0",
#     ),
#     "flip_3d_axis1": PlatonicSolidGroup(
#         torch.stack([
#             torch.eye(3, dtype=torch.float32),
#             torch.diag(torch.tensor([1, -1, 1], dtype=torch.float32)),
#         ]),
#         "flip_3d_axis1",
#     ),
#     "flip_3d_axis2": PlatonicSolidGroup(
#         torch.stack([
#             torch.eye(3, dtype=torch.float32),
#             torch.diag(torch.tensor([1, 1, -1], dtype=torch.float32)),
#         ]),
#         "flip_3d_axis2",
#     ),
# })

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


# # Example usage:
# flop_2d_x = PLATONIC_GROUPS["flop_2d_1"]  # Reflection across x-axis (flips y-coordinate)
# flop_2d_y = PLATONIC_GROUPS["flop_2d_2"]  # Reflection across y-axis (flips x-coordinate)
# flop_3d_x = PLATONIC_GROUPS["flop_3d_1"]  # Reflection across the YZ-plane (flips x-coordinate)
# print("Flop across 2D X-axis elements:\n", flop_2d_x.elements)
# print("\nFlop across 2D Y-axis elements:\n", flop_2d_y.elements)
# print("\nFlop across 3D X-axis elements:\n", flop_3d_x.elements)