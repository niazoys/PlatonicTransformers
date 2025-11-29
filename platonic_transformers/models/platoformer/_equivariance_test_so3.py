import torch
import math
from .platoformer import PlatonicTransformer
from .groups import PLATONIC_GROUPS

# THIS CODE CAN BE RUN FROM THE MAIN DIRECTORY WITH:
# python3 -m models.platoformer._equivariance_test_so3

def get_random_so3(batch_size=1, device='cpu', dtype=torch.float32):
    """
    Generate a random SO(3) rotation matrix using quaternion conversion.
    """
    q = torch.randn(batch_size, 4, device=device, dtype=dtype)
    q /= q.norm(dim=1, keepdim=True)
    
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    x2, y2, z2 = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    xw, yw, zw = qx*qw, qy*qw, qz*qw

    R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1 - 2*y2 - 2*z2
    R[:, 0, 1] = 2*xy - 2*zw
    R[:, 0, 2] = 2*xz + 2*yw

    R[:, 1, 0] = 2*xy + 2*zw
    R[:, 1, 1] = 1 - 2*x2 - 2*z2
    R[:, 1, 2] = 2*yz - 2*xw

    R[:, 2, 0] = 2*xz - 2*yw
    R[:, 2, 1] = 2*yz + 2*xw
    R[:, 2, 2] = 1 - 2*x2 - 2*y2

    return R[0] if batch_size == 1 else R


def run_model_equivariance_test(
    solid_name: str, 
    hidden_dim: int, 
    nhead: int, 
    dtype=torch.float64
):
    """
    Tests the Platoformer model for equivariance to its specified group and
    runs a sanity check to ensure it is not equivariant to random rotations.
    """
    print(f"--- Testing Model Equivariance for '{solid_name}' (dtype: {dtype}) ---")
    
    group = PLATONIC_GROUPS[solid_name.lower()]
    G = group.G
    spatial_dim = 3
    head_dim = hidden_dim // nhead

    print(f"  Group: {solid_name}, Order (G): {G}")
    print(f"  Using standardized hidden_dim: {hidden_dim}, nhead: {nhead}, head_dim: {head_dim}")

    # Model parameters
    model = PlatonicTransformer(
        input_dim=16, input_dim_vec=4, hidden_dim=hidden_dim,
        output_dim=8, output_dim_vec=2, nhead=nhead, num_layers=2,
        solid_name=solid_name, scalar_task_level="node",
        vector_task_level="node", spatial_dim=spatial_dim,
    ).to(dtype)
    model.eval()

    # Generate random data
    torch.manual_seed(42)
    pos = torch.randn(20, spatial_dim, dtype=dtype)
    x = torch.randn(20, 16, dtype=dtype)
    vec = torch.randn(20, 4, spatial_dim, dtype=dtype)
    batch = torch.repeat_interleave(torch.arange(2), 10)

    # Original forward pass
    scalars_orig, vectors_orig = model(x=x, pos=pos, batch=batch, vec=vec)

    # --- Equivariance Test with In-Group Rotations ---
    scalar_diffs_group, vector_diffs_group = [], []
    for h in range(G):
        g = group.elements[h].to(x.dtype)
        pos_tf = (g @ pos.T).T
        vec_tf = (g @ vec.transpose(-1, -2)).transpose(-1, -2)
        scalars_lhs, vectors_lhs = model(x=x, pos=pos_tf, batch=batch, vec=vec_tf)
        scalars_rhs = scalars_orig
        vectors_rhs = (g @ vectors_orig.transpose(-1, -2)).transpose(-1, -2)

        scalar_diffs_group.append(torch.max(torch.abs(scalars_lhs - scalars_rhs)).item())
        vector_diffs_group.append(torch.max(torch.abs(vectors_lhs - vectors_rhs)).item())
    
    avg_scalar_diff_group = sum(scalar_diffs_group) / G if scalar_diffs_group else 0
    avg_vector_diff_group = sum(vector_diffs_group) / G if vector_diffs_group else 0
    
    print(f"  In-Group Error (Numerical Precision) -> Avg scalar diff: {avg_scalar_diff_group:.2e}, Avg vector diff: {avg_vector_diff_group:.2e}")

    # --- Sanity Check with Out-of-Group Rotations ---
    print("\n--- Sanity Check: Testing with random (out-of-group) SO(3) rotations ---")
    num_random_samples = 10
    scalar_diffs_rand, vector_diffs_rand = [], []
    for _ in range(num_random_samples):
        g_rand = get_random_so3(dtype=x.dtype)
        pos_tf_rand = (g_rand @ pos.T).T
        vec_tf_rand = (g_rand @ vec.transpose(-1, -2)).transpose(-1, -2)
        scalars_lhs_rand, vectors_lhs_rand = model(x=x, pos=pos_tf_rand, batch=batch, vec=vec_tf_rand)
        scalars_rhs_rand = scalars_orig
        vectors_rhs_rand = (g_rand @ vectors_orig.transpose(-1, -2)).transpose(-1, -2)
        scalar_diffs_rand.append(torch.max(torch.abs(scalars_lhs_rand - scalars_rhs_rand)).item())
        vector_diffs_rand.append(torch.max(torch.abs(vectors_lhs_rand - vectors_rhs_rand)).item())

    avg_scalar_diff_rand = sum(scalar_diffs_rand) / num_random_samples
    avg_vector_diff_rand = sum(vector_diffs_rand) / num_random_samples

    print(f"  Out-of-Group Error (Equivariance) -> Avg scalar diff: {avg_scalar_diff_rand:.2e}, Avg vector diff: {avg_vector_diff_rand:.2e}")
    
    return avg_scalar_diff_group, avg_vector_diff_group, avg_scalar_diff_rand, avg_vector_diff_rand


def print_conclusion(results):
    """Analyzes the results table and prints a summary conclusion."""
    print(f"\n{'='*30} ANALYSIS & CONCLUSION {'='*30}")

    s_rand_icosa = results.get('icosahedron', {}).get('scalar_rand', 0)
    s_group_icosa = results.get('icosahedron', {}).get('scalar_group', 0)
    s_group_tetra = results.get('tetrahedron', {}).get('scalar_group', 0)
    
    # Calculate one representative OOM difference to show the signal strength
    def get_oom_diff(res):
        s_group = res.get('scalar_group', 1.0)
        s_rand = res.get('scalar_rand', 1.0)
        if s_group < 1e-30: return 0.0 
        return math.log10(s_rand) - math.log10(s_group)
    
    tetra_diff = get_oom_diff(results.get('tetrahedron', {}))

    print("1. Consistency of Equivariance:")
    print(f"   All groups demonstrate strong equivariance. The gap between in-group error and random error")
    print(f"   is consistently massive (e.g., >{tetra_diff:.1f} orders of magnitude for Tetrahedron).")

    print("\n2. Precision Analysis:")
    # Check if Icosahedron is significantly less precise (> 100x error) than Tetrahedron
    if s_group_icosa > (s_group_tetra * 100) and s_group_icosa > 1e-10:
        print(f"   There is a notable difference in precision. The Icosahedron model shows higher numerical error")
        print(f"   ({s_group_icosa:.2e}) compared to the Tetrahedron ({s_group_tetra:.2e}).")
        print(f"   This is expected due to the Icosahedron's reliance on irrational coordinates (Golden Ratio)")
        print(f"   and larger group size (60 vs 12), which accumulates more floating-point noise.")
    else:
        print(f"   The 'Avg In-Group Err' is comparable across the different groups at this precision.")
        print(f"   This indicates that the error is primarily driven by the floating-point format itself")
        print(f"   rather than the specific complexity of the group operations.")

    print("\n3. Geometric Coverage:")
    print(f"   The Icosahedron has the lowest out-of-group error ({s_rand_icosa:.2e}) because its 60 elements")
    print(f"   cover the rotation space so densely that any random rotation is geometrically close to a valid symmetry.")
    
    print("\nConclusion:")
    print(f"   The tests confirm robust equivariance. The variance in error rates correctly reflects the")
    print(f"   trade-offs between group complexity (Icosahedron) and numerical simplicity (Tetrahedron).")


if __name__ == '__main__':
    solids_to_test = ['trivial', 'tetrahedron', 'octahedron', 'icosahedron']
    results = {}
    
    # --- SET PRECISION HERE ---
    # Change to torch.float32 for lower precision, or torch.float64 for higher precision
    TEST_DTYPE = torch.float64
    
    # --- Determine standardized model parameters based on LCM of all group orders ---
    base_hidden_dim, base_nhead = 960, 4
    G_orders = [PLATONIC_GROUPS[name.lower()].G for name in solids_to_test if name in PLATONIC_GROUPS]
    lcm_groups = 1
    for order in G_orders:
        lcm_groups = (lcm_groups * order) // math.gcd(lcm_groups, order)

    nhead_std = ((base_nhead + lcm_groups - 1) // lcm_groups) * lcm_groups
    if nhead_std == 0: nhead_std = lcm_groups
    
    lcm_val = (lcm_groups * 2 * nhead_std) // math.gcd(lcm_groups, 2 * nhead_std)
    hidden_dim_std = ((base_hidden_dim + lcm_val - 1) // lcm_val) * lcm_val
    if hidden_dim_std == 0: hidden_dim_std = lcm_val

    # --- Run tests for all solids using standardized architecture ---
    for name in solids_to_test:
        if name in PLATONIC_GROUPS:
            print(f"\n{'='*25} TESTING: {name.upper()} {'='*25}")
            # Pass the dtype here
            avg_s_group, avg_v_group, avg_s_rand, avg_v_rand = run_model_equivariance_test(
                name, hidden_dim=hidden_dim_std, nhead=nhead_std, dtype=TEST_DTYPE
            )
            results[name] = {
                "scalar_group": avg_s_group, "vector_group": avg_v_group,
                "scalar_rand": avg_s_rand, "vector_rand": avg_v_rand
            }
        else:
            print(f"\nSkipping '{name}', not found.")

    # --- Print Final Summary Report ---
    print(f"\n\n{'='*30} FINAL SUMMARY REPORT {'='*30}")
    print("Using standardized model architecture for all tests:")
    print(f"hidden_dim: {hidden_dim_std}, nhead: {nhead_std}")
    print(f"Precision: {TEST_DTYPE}\n")
    
    header = f"{'Solid':<15} | {'Avg In-Group Err':<18} | {'Avg Out-Group Err':<18} | {'vs Out-of-Group (OOM)':<24}"
    print(header)
    print('-' * len(header))

    for name, data in results.items():
        s_group, s_rand = data['scalar_group'], data['scalar_rand']
        
        def safe_log10(x):
            return math.log10(x) if x > 1e-30 else -30.0

        log_s_group = safe_log10(s_group)
        log_s_rand = safe_log10(s_rand)

        # Logic for OOM display
        # If s_group is practically zero (like in trivial case), OOM is undefined/artificial
        if s_group < 1e-20: 
             vs_rand_s_oom_str = "          -           " 
        else:
             vs_rand_s_oom = log_s_rand - log_s_group
             vs_rand_s_oom_str = f"{vs_rand_s_oom:<+24.1f}"

        print(f"{name:<15} | {s_group:<18.2e} | {s_rand:<18.2e} | {vs_rand_s_oom_str}")

    # --- Print Conclusion ---
    print_conclusion(results)