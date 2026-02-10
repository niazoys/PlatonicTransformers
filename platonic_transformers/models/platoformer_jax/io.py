"""
Input/Output utilities for JAX Platoformer.

Contains functions for lifting features to group representation, pooling,
and converting between sparse and dense formats.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from platonic_transformers.models.platoformer_jax.groups import PlatonicSolidGroup
from platonic_transformers.models.platoformer_jax.utils import segment_sum, segment_mean


def lift_scalars(x: jnp.ndarray, group: PlatonicSolidGroup) -> jnp.ndarray:
    """
    Lift scalar features to group representation by replication.
    
    Args:
        x: Scalar features of shape (N, C) for graph mode or (B, N, C) for dense mode
        group: Platonic group
        
    Returns:
        Lifted features of shape (N, G, C) or (B, N, G, C)
    """
    if x.ndim == 2:  # graph mode: (N, C)
        return jnp.broadcast_to(x[:, None, :], (x.shape[0], group.G, x.shape[1]))
    elif x.ndim == 3:  # dense mode: (B, N, C)
        return jnp.broadcast_to(x[:, :, None, :], (x.shape[0], x.shape[1], group.G, x.shape[2]))
    else:
        raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")


def lift_vectors(x: jnp.ndarray, group: PlatonicSolidGroup) -> jnp.ndarray:
    """
    Lift vector features to group representation by applying group rotations.
    
    Args:
        x: Vector features of shape (..., C, D) where D is spatial dimension
        group: Platonic group
        
    Returns:
        Lifted features of shape (..., G, C*D)
    """
    frames = group.get_elements_jax()  # (G, D, D)
    # Apply inverse transpose (which for orthogonal matrices is just the matrix itself)
    # gji -> transposed frame
    result = jnp.einsum('gji,...cj->...gci', frames, x)
    return result.reshape(*x.shape[:-2], group.G, -1)


def readout_scalars(x: jnp.ndarray, group: PlatonicSolidGroup) -> jnp.ndarray:
    """
    Readout scalars by averaging over group dimension.
    
    Args:
        x: Features of shape (..., G, C)
        group: Platonic group
        
    Returns:
        Scalar features of shape (..., C)
    """
    return x.mean(axis=-2)


def readout_vectors(x: jnp.ndarray, group: PlatonicSolidGroup) -> jnp.ndarray:
    """
    Readout vectors by applying inverse group rotations and averaging.
    
    Args:
        x: Features of shape (..., G, C*D) where D is spatial dimension
        group: Platonic group
        
    Returns:
        Vector features of shape (..., C, D)
    """
    # Unflatten: (..., G, C*D) -> (..., G, C, D)
    x = x.reshape(*x.shape[:-1], -1, group.dim)
    frames = group.get_elements_jax()  # (G, D, D)
    # Apply frame rotation and average
    result = jnp.einsum('gij,...gcj->...ci', frames, x) / group.G
    return result


def lift(scalars: Optional[jnp.ndarray], 
         vectors: Optional[jnp.ndarray], 
         group: PlatonicSolidGroup) -> jnp.ndarray:
    """
    Lift scalar and vector features to group representation.
    
    Args:
        scalars: Optional scalar features
        vectors: Optional vector features
        group: Platonic group
        
    Returns:
        Combined lifted features of shape (..., G*C)
    """
    x_list = []
    if scalars is not None:
        x_list.append(lift_scalars(scalars, group))
    if vectors is not None:
        x_list.append(lift_vectors(vectors, group))
    
    if not x_list:
        raise ValueError("At least one of scalars or vectors must be provided")
    
    # Concatenate along channel dimension and flatten group dimension
    combined = jnp.concatenate(x_list, axis=-1)
    return combined.reshape(*combined.shape[:-2], -1)


def to_scalars_vectors(x: jnp.ndarray, 
                       num_scalars: int, 
                       num_vectors: int, 
                       group: PlatonicSolidGroup) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert group features back to scalars and vectors.
    
    Args:
        x: Features of shape (..., G*C) where C = num_scalars + num_vectors*D
        num_scalars: Number of scalar channels
        num_vectors: Number of vector channels
        group: Platonic group
        
    Returns:
        Tuple of (scalars, vectors) with shapes (..., num_scalars) and (..., num_vectors, D)
    """
    # Unflatten: (..., G*C) -> (..., G, C)
    x = x.reshape(*x.shape[:-1], group.G, -1)
    
    # Split scalar and vector parts
    x_scalars, x_vectors = jnp.split(x, [num_scalars], axis=-1)
    
    # Readout
    scalars = readout_scalars(x_scalars, group) if num_scalars > 0 else jnp.zeros((*x.shape[:-2], 0))
    vectors = readout_vectors(x_vectors, group) if num_vectors > 0 else jnp.zeros((*x.shape[:-2], 0, group.dim))
    
    return scalars, vectors


def to_dense_and_mask(
    x: Optional[jnp.ndarray],
    vec: Optional[jnp.ndarray],
    pos: jnp.ndarray,
    batch: Optional[jnp.ndarray]
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Convert sparse graph input to dense, padded tensors.
    
    Args:
        x: Node features of shape (N, C) for sparse or (B, N, C) for dense
        vec: Vector features of shape (N, C, D) for sparse or (B, N, C, D) for dense
        pos: Node positions of shape (N, D) for sparse or (B, N, D) for dense
        batch: Batch indices of shape (N,) for sparse, None for dense
        
    Returns:
        Tuple of (dense_x, dense_vec, dense_pos, mask)
    """
    if x is None and vec is None:
        raise ValueError("At least one of x or vec must be provided.")
    
    if batch is None:  # Already dense
        if x is not None and x.ndim != 3:
            raise ValueError("If batch is None, x must be 3D [B, N, C]")
        if pos.ndim != 3:
            raise ValueError("If batch is None, pos must be 3D [B, N, D]")
        
        B, N = (x.shape[0], x.shape[1]) if x is not None else (pos.shape[0], pos.shape[1])
        mask = jnp.ones((B, N), dtype=jnp.bool_)
        return x, vec, pos, mask
    
    # Sparse to dense conversion
    num_graphs = int(batch.max()) + 1
    
    # Count nodes per graph
    counts = jnp.bincount(batch, length=num_graphs)
    max_nodes = int(counts.max())
    
    # Create dense tensors using padding
    def sparse_to_dense(sparse_tensor, fill_value=0.0):
        if sparse_tensor is None:
            return None
        
        feature_shape = sparse_tensor.shape[1:]
        dense_shape = (num_graphs, max_nodes) + feature_shape
        
        # Create indices for scatter
        node_indices = jnp.zeros(sparse_tensor.shape[0], dtype=jnp.int32)
        
        # This is a simplified version - in practice you'd want to properly
        # compute the local indices within each graph
        # For now, use a simple approach
        def compute_local_indices(batch):
            # Compute local index within each graph
            local_idx = jnp.zeros_like(batch)
            for i in range(num_graphs):
                mask = batch == i
                graph_nodes = jnp.cumsum(mask) - 1
                local_idx = jnp.where(mask, graph_nodes, local_idx)
            return local_idx
        
        local_indices = compute_local_indices(batch)
        
        # Build dense tensor
        dense = jnp.zeros(dense_shape, dtype=sparse_tensor.dtype)
        dense = dense.at[batch, local_indices].set(sparse_tensor)
        
        return dense
    
    dense_x = sparse_to_dense(x)
    dense_vec = sparse_to_dense(vec)
    dense_pos = sparse_to_dense(pos)
    
    # Create mask
    mask = jnp.zeros((num_graphs, max_nodes), dtype=jnp.bool_)
    for i in range(num_graphs):
        mask = mask.at[i, :counts[i]].set(True)
    
    return dense_x, dense_vec, dense_pos, mask


def pool(
    x: jnp.ndarray,
    batch: Optional[jnp.ndarray],
    mask: Optional[jnp.ndarray] = None,
    avg_num_nodes: Optional[float] = None,
    dense_mode: bool = False,
    mean_aggregation: bool = True
) -> jnp.ndarray:
    """
    Pool node features to graph-level features.
    
    Args:
        x: Node features
        batch: Batch indices (graph mode)
        mask: Attention mask (dense mode)
        avg_num_nodes: Average number of nodes for scaling
        dense_mode: Whether in dense mode
        mean_aggregation: Whether to compute mean (vs. sum scaled by avg_num_nodes)
        
    Returns:
        Pooled features
    """
    if dense_mode:
        # Dense mode: x is [B, N_max, hidden_dim]
        if mask is not None:
            x = x * mask[..., None]
        x = x.sum(axis=1)  # [B, hidden_dim]
        
        if mean_aggregation and mask is not None:
            num_nodes = mask.sum(axis=1, keepdims=True).astype(jnp.float32)
            num_nodes = jnp.maximum(num_nodes, 1.0)
        else:
            num_nodes = avg_num_nodes if avg_num_nodes is not None else 1.0
        
        x = x / num_nodes
    else:
        # Graph mode: x is [N, hidden_dim]
        assert batch is not None, "batch must be provided in graph mode"
        num_graphs = int(jnp.max(batch)) + 1
        
        if mean_aggregation:
            x = segment_mean(x, batch, num_graphs)
        else:
            x = segment_sum(x, batch, num_graphs)
            if avg_num_nodes is not None and avg_num_nodes != 1.0:
                x = x / avg_num_nodes
    
    return x
