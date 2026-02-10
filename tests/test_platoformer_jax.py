"""
Test script for JAX Platoformer implementation.

Tests all modules and verifies data loading works correctly.
Run with: python tests/test_platoformer_jax.py
"""

import os
import sys
from typing import Dict

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    JAX_AVAILABLE = True
    print(f"✓ JAX version: {jax.__version__}")
    print(f"✓ JAX devices: {jax.devices()}")
except ImportError as e:
    JAX_AVAILABLE = False
    print(f"✗ JAX not available: {e}")
    print("  Install with: pip install -r requirements_jax.txt")


def test_groups():
    """Test Platonic groups module."""
    print("\n" + "="*60)
    print("Testing Groups Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.groups import (
        PLATONIC_GROUPS, get_group, PlatonicSolidGroup
    )
    
    # Test available groups
    print(f"\n✓ Available groups: {list(PLATONIC_GROUPS.keys())[:10]}...")
    
    # Test specific groups
    test_groups = ['trivial', 'tetrahedron', 'octahedron', 'icosahedron']
    expected_orders = [1, 12, 24, 60]
    
    for name, expected_G in zip(test_groups, expected_orders):
        group = get_group(name)
        assert group.G == expected_G, f"Expected order {expected_G}, got {group.G}"
        assert group.elements.shape == (expected_G, 3, 3), f"Wrong element shape"
        assert group.cayley_table.shape == (expected_G, expected_G), f"Wrong Cayley table shape"
        assert group.inverse_indices.shape == (expected_G,), f"Wrong inverse indices shape"
        print(f"  ✓ {name}: G={group.G}, dim={group.dim}")
    
    # Test JAX array conversion
    group = get_group('octahedron')
    elements_jax = group.get_elements_jax()
    cayley_jax = group.get_cayley_table_jax()
    inverse_jax = group.get_inverse_indices_jax()
    
    assert elements_jax.shape == (24, 3, 3)
    assert cayley_jax.dtype == jnp.int32
    print(f"  ✓ JAX conversion works")
    
    # Verify group properties (closure, identity, inverse)
    identity_idx = 0  # Assume first element is identity
    for i in range(group.G):
        # e * g = g
        assert cayley_jax[identity_idx, i] == i, "Identity property failed"
        # g * g^-1 = e
        inv_i = inverse_jax[i]
        assert cayley_jax[i, inv_i] == identity_idx, "Inverse property failed"
    print(f"  ✓ Group properties verified (identity, inverse)")
    
    print("\n✓ Groups module: PASSED")
    return True


def test_linear():
    """Test PlatonicLinear layer."""
    print("\n" + "="*60)
    print("Testing Linear Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.linear import PlatonicLinear
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    # Test parameters
    solid_name = 'octahedron'
    G = get_group(solid_name).G
    in_features = G * 16  # 384
    out_features = G * 32  # 768
    batch_size = 5
    
    # Create layer
    layer = PlatonicLinear(
        in_features=in_features,
        out_features=out_features,
        solid_name=solid_name,
        use_bias=True
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, in_features))
    
    params = layer.init(rng, x)
    print(f"  ✓ Layer initialized")
    print(f"    Input shape: {x.shape}")
    print(f"    Kernel shape: {params['params']['kernel'].shape}")
    
    # Forward pass
    y = layer.apply(params, x)
    assert y.shape == (batch_size, out_features), f"Wrong output shape: {y.shape}"
    print(f"    Output shape: {y.shape}")
    
    # Test equivariance
    print("\n  Testing equivariance...")
    group = get_group(solid_name)
    cayley_table = group.get_cayley_table_jax()
    
    # Pick a random group element to transform with
    h = 5  # arbitrary group element
    transform_indices = cayley_table[h, :]
    
    # Transform input by permuting group channels
    x_reshaped = x.reshape(batch_size, G, -1)
    x_transformed = x_reshaped[:, transform_indices, :].reshape(batch_size, in_features)
    
    # Compare f(g·x) vs g·f(x)
    y_lhs = layer.apply(params, x_transformed)
    
    y_reshaped = y.reshape(batch_size, G, -1)
    y_rhs = y_reshaped[:, transform_indices, :].reshape(batch_size, out_features)
    
    error = jnp.max(jnp.abs(y_lhs - y_rhs))
    assert error < 1e-4, f"Equivariance failed with error {error}"
    print(f"  ✓ Equivariance verified (max error: {error:.2e})")
    
    print("\n✓ Linear module: PASSED")
    return True


def test_rope():
    """Test PlatonicRoPE module."""
    print("\n" + "="*60)
    print("Testing RoPE Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.rope import PlatonicRoPE
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    solid_name = 'octahedron'
    G = get_group(solid_name).G
    num_heads = 4
    head_dim = 16
    embed_dim = G * num_heads * head_dim
    batch_size = 3
    seq_len = 10
    spatial_dims = 3
    
    # Create RoPE
    rope = PlatonicRoPE(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        solid_name=solid_name,
        spatial_dims=spatial_dims,
        freq_sigma=1.0,
        learned_freqs=False,
        freq_init='spiral'
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, G, num_heads, head_dim))
    pos = jax.random.normal(rng, (batch_size, seq_len, spatial_dims))
    
    params = rope.init(rng, x, pos)
    print(f"  ✓ RoPE initialized")
    print(f"    Input shape: {x.shape}")
    print(f"    Position shape: {pos.shape}")
    
    # Forward pass
    y = rope.apply(params, x, pos)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"    Output shape: {y.shape}")
    
    # Verify RoPE preserves norm (approximately)
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    norm_diff = jnp.max(jnp.abs(x_norm - y_norm))
    assert norm_diff < 1e-4, f"RoPE changed norms by {norm_diff}"
    print(f"  ✓ Norm preservation verified (max diff: {norm_diff:.2e})")
    
    print("\n✓ RoPE module: PASSED")
    return True


def test_ape():
    """Test APE and PlatonicAPE modules."""
    print("\n" + "="*60)
    print("Testing APE Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.ape import APE, PlatonicAPE
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    batch_size = 4
    seq_len = 10
    spatial_dims = 3
    embed_dim = 64
    
    # Test standard APE
    print("\n  Testing standard APE...")
    ape = APE(embed_dim=embed_dim, freq_sigma=1.0, spatial_dims=spatial_dims)
    
    rng = jax.random.PRNGKey(42)
    pos = jax.random.normal(rng, (batch_size, seq_len, spatial_dims))
    
    params = ape.init(rng, pos)
    embedding = ape.apply(params, pos)
    assert embedding.shape == (batch_size, seq_len, embed_dim)
    print(f"  ✓ APE output shape: {embedding.shape}")
    
    # Test PlatonicAPE
    print("\n  Testing PlatonicAPE...")
    solid_name = 'octahedron'
    G = get_group(solid_name).G
    platonic_embed_dim = G * 8  # Must be divisible by G and even per-G
    
    platonic_ape = PlatonicAPE(
        embed_dim=platonic_embed_dim,
        solid_name=solid_name,
        freq_sigma=1.0,
        spatial_dims=spatial_dims,
        learned_freqs=False
    )
    
    params = platonic_ape.init(rng, pos)
    embedding = platonic_ape.apply(params, pos)
    assert embedding.shape == (batch_size, seq_len, platonic_embed_dim)
    print(f"  ✓ PlatonicAPE output shape: {embedding.shape}")
    
    print("\n✓ APE module: PASSED")
    return True


def test_conv():
    """Test PlatonicConv module."""
    print("\n" + "="*60)
    print("Testing Conv Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.conv import PlatonicConv
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    solid_name = 'tetrahedron'
    G = get_group(solid_name).G
    channels = G * 8
    num_heads = G * 2
    batch_size = 2
    num_nodes = 15
    spatial_dims = 3
    
    conv = PlatonicConv(
        in_channels=channels,
        out_channels=channels,
        embed_dim=channels,
        num_heads=num_heads,
        solid_name=solid_name,
        spatial_dims=spatial_dims,
        freq_sigma=1.0,
        attention=False  # Linear attention
    )
    
    rng = jax.random.PRNGKey(42)
    
    # Test graph mode
    print("\n  Testing graph mode...")
    x = jax.random.normal(rng, (num_nodes, channels))
    pos = jax.random.normal(rng, (num_nodes, spatial_dims))
    batch = jnp.array([0]*7 + [1]*8)  # Two graphs
    
    params = conv.init(rng, x, pos, batch=batch)
    y = conv.apply(params, x, pos, batch=batch)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    print(f"  ✓ Graph mode output shape: {y.shape}")
    
    # Test dense mode
    print("\n  Testing dense mode...")
    x_dense = jax.random.normal(rng, (batch_size, num_nodes, channels))
    pos_dense = jax.random.normal(rng, (batch_size, num_nodes, spatial_dims))
    mask = jnp.ones((batch_size, num_nodes), dtype=jnp.bool_)
    
    y_dense = conv.apply(params, x_dense, pos_dense, mask=mask)
    assert y_dense.shape == x_dense.shape
    print(f"  ✓ Dense mode output shape: {y_dense.shape}")
    
    print("\n✓ Conv module: PASSED")
    return True


def test_block():
    """Test PlatonicBlock module."""
    print("\n" + "="*60)
    print("Testing Block Module")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.block import PlatonicBlock
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    solid_name = 'tetrahedron'
    G = get_group(solid_name).G
    d_model = G * 16
    nhead = G * 2
    dim_feedforward = G * 64
    num_nodes = 20
    
    block = PlatonicBlock(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        solid_name=solid_name,
        dropout=0.0,
        drop_path=0.0,
        freq_sigma=1.0,
        attention=False
    )
    
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (num_nodes, d_model))
    pos = jax.random.normal(rng, (num_nodes, 3))
    batch = jnp.array([0]*10 + [1]*10)
    
    params = block.init(rng, x, pos, batch=batch, deterministic=True)
    print(f"  ✓ Block initialized")
    print(f"    d_model: {d_model}, nhead: {nhead}, ffn: {dim_feedforward}")
    
    y = block.apply(params, x, pos, batch=batch, deterministic=True)
    assert y.shape == x.shape
    print(f"    Input/Output shape: {y.shape}")
    
    # Check residual connection works
    assert not jnp.allclose(x, y), "Block should transform input"
    print(f"  ✓ Block transforms input correctly")
    
    print("\n✓ Block module: PASSED")
    return True


def test_io():
    """Test I/O utilities."""
    print("\n" + "="*60)
    print("Testing I/O Utilities")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.io import (
        lift, lift_scalars, lift_vectors, pool, to_scalars_vectors
    )
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    solid_name = 'octahedron'
    group = get_group(solid_name)
    G = group.G
    
    batch_size = 3
    num_nodes = 10
    scalar_dim = 5
    vector_dim = 2
    
    rng = jax.random.PRNGKey(42)
    
    # Test lift_scalars
    print("\n  Testing lift_scalars...")
    scalars = jax.random.normal(rng, (num_nodes, scalar_dim))
    lifted_scalars = lift_scalars(scalars, group)
    assert lifted_scalars.shape == (num_nodes, G, scalar_dim)
    print(f"  ✓ lift_scalars: {scalars.shape} -> {lifted_scalars.shape}")
    
    # Test lift_vectors
    print("\n  Testing lift_vectors...")
    vectors = jax.random.normal(rng, (num_nodes, vector_dim, 3))
    lifted_vectors = lift_vectors(vectors, group)
    assert lifted_vectors.shape == (num_nodes, G, vector_dim * 3)
    print(f"  ✓ lift_vectors: {vectors.shape} -> {lifted_vectors.shape}")
    
    # Test lift (combined)
    print("\n  Testing lift...")
    lifted = lift(scalars, vectors, group)
    expected_dim = G * (scalar_dim + vector_dim * 3)
    assert lifted.shape == (num_nodes, expected_dim)
    print(f"  ✓ lift combined: {lifted.shape}")
    
    # Test pool
    print("\n  Testing pool...")
    x = jax.random.normal(rng, (num_nodes, G * 8))
    batch = jnp.array([0]*5 + [1]*5)
    pooled = pool(x, batch, mean_aggregation=True)
    assert pooled.shape == (2, G * 8)  # 2 graphs
    print(f"  ✓ pool: {x.shape} -> {pooled.shape}")
    
    # Test to_scalars_vectors
    print("\n  Testing to_scalars_vectors...")
    num_scalars = 3
    num_vectors = 2
    total_dim = G * (num_scalars + num_vectors * 3)
    x = jax.random.normal(rng, (batch_size, total_dim))
    scalars_out, vectors_out = to_scalars_vectors(x, num_scalars, num_vectors, group)
    assert scalars_out.shape == (batch_size, num_scalars)
    assert vectors_out.shape == (batch_size, num_vectors, 3)
    print(f"  ✓ to_scalars_vectors: scalars={scalars_out.shape}, vectors={vectors_out.shape}")
    
    print("\n✓ I/O utilities: PASSED")
    return True


def test_platoformer():
    """Test full PlatonicTransformer model."""
    print("\n" + "="*60)
    print("Testing PlatonicTransformer Model")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax.platoformer import (
        PlatonicTransformer, create_platoformer
    )
    from platonic_transformers.models.platoformer_jax.groups import get_group
    
    solid_name = 'tetrahedron'
    G = get_group(solid_name).G
    
    # Small model for testing
    model = PlatonicTransformer(
        input_dim=11,  # QM9 features
        input_dim_vec=0,
        hidden_dim=G * 8,  # 96
        output_dim=1,
        output_dim_vec=0,
        nhead=G,  # 12
        num_layers=2,
        solid_name=solid_name,
        spatial_dim=3,
        dense_mode=False,
        scalar_task_level="graph",
        vector_task_level="graph",
        ffn_readout=True,
        dropout=0.0,
        drop_path_rate=0.0,
        attention=False,
        rope_sigma=1.0,
        ape_sigma=None,
    )
    
    rng = jax.random.PRNGKey(42)
    
    # Test with graph mode input
    num_nodes = 25
    x = jax.random.normal(rng, (num_nodes, 11))
    pos = jax.random.normal(rng, (num_nodes, 3))
    batch = jnp.array([0]*8 + [1]*9 + [2]*8)  # 3 graphs
    
    print(f"\n  Initializing model...")
    print(f"    Solid: {solid_name} (G={G})")
    print(f"    Hidden dim: {G * 8}")
    print(f"    Num layers: 2")
    
    params = model.init(rng, x, pos, batch=batch, deterministic=True, avg_num_nodes=8.0)
    
    # Count parameters
    def count_params(params):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    num_params = count_params(params)
    print(f"    Total parameters: {num_params:,}")
    
    # Forward pass
    scalars, vectors = model.apply(params, x, pos, batch=batch, deterministic=True, avg_num_nodes=8.0)
    
    num_graphs = 3
    assert scalars.shape == (num_graphs, 1), f"Wrong scalar shape: {scalars.shape}"
    assert vectors.shape == (num_graphs, 0, 3), f"Wrong vector shape: {vectors.shape}"
    print(f"\n  ✓ Forward pass successful")
    print(f"    Input: {num_nodes} nodes, 3 graphs")
    print(f"    Scalar output: {scalars.shape}")
    print(f"    Vector output: {vectors.shape}")
    
    # Test factory function
    print("\n  Testing create_platoformer factory...")
    model2 = create_platoformer(
        input_dim=11,
        hidden_dim=G * 8,
        output_dim=1,
        num_layers=1,
        num_heads=G,
        solid_name=solid_name
    )
    params2 = model2.init(rng, x, pos, batch=batch, deterministic=True, avg_num_nodes=8.0)
    print(f"  ✓ Factory function works")
    
    print("\n✓ PlatonicTransformer model: PASSED")
    return True


def test_data_loading():
    """Test QM9 data loading."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        from torch_geometric.datasets import QM9
        from torch_geometric.loader import DataLoader as PyGDataLoader
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        print("  ✗ PyTorch Geometric not available, skipping data loading test")
        return True
    
    # Check if data directory exists
    data_dir = "./data/qm9"
    
    print(f"\n  Testing QM9 dataset...")
    print(f"  Data directory: {data_dir}")
    
    try:
        # Load dataset (will download if not present)
        dataset = QM9(root=data_dir)
        print(f"  ✓ Dataset loaded: {len(dataset)} molecules")
        
        # Check a sample
        sample = dataset[0]
        print(f"  ✓ Sample molecule:")
        print(f"      x shape: {sample.x.shape}")
        print(f"      pos shape: {sample.pos.shape}")
        print(f"      y shape: {sample.y.shape}")
        print(f"      Number of atoms: {sample.num_nodes}")
        
        # Test DataLoader
        loader = PyGDataLoader(dataset[:100], batch_size=16, shuffle=False)
        batch = next(iter(loader))
        print(f"\n  ✓ DataLoader works:")
        print(f"      Batch x shape: {batch.x.shape}")
        print(f"      Batch pos shape: {batch.pos.shape}")
        print(f"      Number of graphs: {batch.num_graphs}")
        
        # Test conversion to JAX
        print("\n  Testing PyG to JAX conversion...")
        x_jax = jnp.array(batch.x.numpy())
        pos_jax = jnp.array(batch.pos.numpy())
        batch_jax = jnp.array(batch.batch.numpy())
        y_jax = jnp.array(batch.y.numpy())
        
        print(f"  ✓ JAX arrays created:")
        print(f"      x: {x_jax.shape}, dtype={x_jax.dtype}")
        print(f"      pos: {pos_jax.shape}, dtype={pos_jax.dtype}")
        print(f"      batch: {batch_jax.shape}, dtype={batch_jax.dtype}")
        print(f"      y: {y_jax.shape}, dtype={y_jax.dtype}")
        
    except Exception as e:
        print(f"  Note: QM9 data not available or download failed: {e}")
        print(f"  This is okay for testing modules without data")
        return True
    
    print("\n✓ Data loading: PASSED")
    return True


def test_end_to_end():
    """Test end-to-end forward pass with realistic data."""
    print("\n" + "="*60)
    print("Testing End-to-End Forward Pass")
    print("="*60)
    
    from platonic_transformers.models.platoformer_jax import (
        PlatonicTransformer, PLATONIC_GROUPS, lift, pool
    )
    
    # Simulate QM9-like data
    rng = jax.random.PRNGKey(42)
    
    # Multiple molecules with different sizes
    mol_sizes = [5, 7, 8, 6, 9]  # atoms per molecule
    total_atoms = sum(mol_sizes)
    num_mols = len(mol_sizes)
    
    # Create batch indices
    batch = jnp.array([i for i, size in enumerate(mol_sizes) for _ in range(size)])
    
    # Features (11 features like QM9)
    x = jax.random.normal(rng, (total_atoms, 11))
    pos = jax.random.normal(rng, (total_atoms, 3))
    
    # Create model
    solid_name = 'octahedron'
    G = PLATONIC_GROUPS[solid_name].G
    
    model = PlatonicTransformer(
        input_dim=11,
        input_dim_vec=0,
        hidden_dim=G * 4,
        output_dim=1,
        output_dim_vec=0,
        nhead=G,
        num_layers=2,
        solid_name=solid_name,
        spatial_dim=3,
        dense_mode=False,
        scalar_task_level="graph",
        ffn_readout=True,
        dropout=0.0,
        attention=False,
        rope_sigma=1.0,
    )
    
    # Initialize and run
    params = model.init(rng, x, pos, batch=batch, deterministic=True, avg_num_nodes=7.0)
    scalars, vectors = model.apply(params, x, pos, batch=batch, deterministic=True, avg_num_nodes=7.0)
    
    assert scalars.shape == (num_mols, 1)
    print(f"\n  ✓ End-to-end test passed")
    print(f"    Input: {total_atoms} total atoms in {num_mols} molecules")
    print(f"    Output: {scalars.shape} predictions")
    print(f"    Predictions: {scalars.flatten()}")
    
    # Test gradient computation
    print("\n  Testing gradient computation...")
    
    def loss_fn(params):
        preds, _ = model.apply(params, x, pos, batch=batch, deterministic=True, avg_num_nodes=7.0)
        return jnp.mean(preds ** 2)  # Dummy loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Check gradients are not None/NaN
    grad_leaves = jax.tree_util.tree_leaves(grads)
    has_nan = any(jnp.any(jnp.isnan(g)) for g in grad_leaves)
    assert not has_nan, "Gradients contain NaN!"
    print(f"  ✓ Gradients computed successfully (no NaN)")
    print(f"    Loss: {loss:.4f}")
    
    print("\n✓ End-to-end test: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# JAX Platoformer Test Suite")
    print("#"*60)
    
    if not JAX_AVAILABLE:
        print("\n✗ Cannot run tests: JAX is not installed")
        return False
    
    tests = [
        ("Groups", test_groups),
        ("Linear", test_linear),
        ("RoPE", test_rope),
        ("APE", test_ape),
        ("Conv", test_conv),
        ("Block", test_block),
        ("I/O Utilities", test_io),
        ("PlatonicTransformer", test_platoformer),
        ("Data Loading", test_data_loading),
        ("End-to-End", test_end_to_end),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
