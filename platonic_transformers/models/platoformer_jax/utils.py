"""
Utility functions for the JAX Platoformer implementation.

Contains scatter operations and other helper functions.
"""

import jax
import jax.numpy as jnp
from typing import Optional


def scatter_add(src: jnp.ndarray, 
                index: jnp.ndarray, 
                dim_size: int,
                axis: int = 0) -> jnp.ndarray:
    """
    Scatter-add operation: accumulates values from src into output at positions specified by index.
    
    Args:
        src: Source tensor to scatter from
        index: Index tensor specifying where to add each element
        dim_size: Size of the output dimension
        axis: Axis along which to scatter (default: 0)
        
    Returns:
        Output tensor with scattered values accumulated
    """
    # Ensure index has the same number of dimensions as src
    while index.ndim < src.ndim:
        index = jnp.expand_dims(index, axis=-1)
    
    # Broadcast index to match src shape
    index = jnp.broadcast_to(index, src.shape)
    
    # Create output shape
    out_shape = list(src.shape)
    out_shape[axis] = dim_size
    
    # Use segment_sum for scatter add
    out = jnp.zeros(out_shape, dtype=src.dtype)
    
    # For 1D case, use simple add.at
    if src.ndim == 1:
        out = out.at[index].add(src)
    else:
        # Flatten all dimensions except the scatter axis
        # Move scatter axis to front
        src_transposed = jnp.moveaxis(src, axis, 0)
        index_transposed = jnp.moveaxis(index, axis, 0)
        
        original_shape = src_transposed.shape[1:]
        flat_src = src_transposed.reshape(src_transposed.shape[0], -1)
        flat_index = index_transposed.reshape(index_transposed.shape[0], -1)
        
        # Create output and scatter
        out_flat = jnp.zeros((dim_size, flat_src.shape[1]), dtype=src.dtype)
        
        # Use vmap over the second axis for vectorized scatter
        def scatter_single_column(src_col, idx_col):
            return jnp.zeros(dim_size, dtype=src.dtype).at[idx_col].add(src_col)
        
        out_flat = jax.vmap(scatter_single_column, in_axes=(1, 1), out_axes=1)(flat_src, flat_index)
        
        # Reshape back
        out = out_flat.reshape(dim_size, *original_shape)
        out = jnp.moveaxis(out, 0, axis)
    
    return out


def scatter_mean(src: jnp.ndarray,
                 index: jnp.ndarray,
                 dim_size: int,
                 axis: int = 0) -> jnp.ndarray:
    """
    Scatter-mean operation: computes mean of values at each index position.
    
    Args:
        src: Source tensor to scatter from  
        index: Index tensor specifying grouping
        dim_size: Size of the output dimension
        axis: Axis along which to scatter (default: 0)
        
    Returns:
        Output tensor with mean values at each index
    """
    # Compute sum
    sum_result = scatter_add(src, index, dim_size, axis)
    
    # Count elements at each index
    ones = jnp.ones_like(src)
    count = scatter_add(ones, index, dim_size, axis)
    
    # Avoid division by zero
    count = jnp.maximum(count, 1.0)
    
    return sum_result / count


def segment_sum(data: jnp.ndarray, 
                segment_ids: jnp.ndarray, 
                num_segments: int) -> jnp.ndarray:
    """
    Computes the sum of segments of a tensor.
    
    This is a cleaner interface for graph aggregation operations.
    
    Args:
        data: Input data tensor of shape [N, ...]
        segment_ids: Integer tensor of shape [N] indicating segment membership
        num_segments: Total number of segments
        
    Returns:
        Tensor of shape [num_segments, ...] with summed values per segment
    """
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)


def segment_mean(data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: int) -> jnp.ndarray:
    """
    Computes the mean of segments of a tensor.
    
    Args:
        data: Input data tensor of shape [N, ...]
        segment_ids: Integer tensor of shape [N] indicating segment membership
        num_segments: Total number of segments
        
    Returns:
        Tensor of shape [num_segments, ...] with mean values per segment
    """
    sums = jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)
    counts = jax.ops.segment_sum(jnp.ones(data.shape[0]), segment_ids, num_segments=num_segments)
    counts = jnp.maximum(counts, 1.0)
    
    # Expand counts to match sums shape
    for _ in range(sums.ndim - 1):
        counts = jnp.expand_dims(counts, axis=-1)
    
    return sums / counts


def drop_path(x: jnp.ndarray, 
              drop_prob: float,
              rng: Optional[jax.random.PRNGKey], 
              training: bool = True,
              scale_by_keep: bool = True) -> jnp.ndarray:
    """
    Drop paths (Stochastic Depth) per sample.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        rng: JAX random key (required if training and drop_prob > 0)
        training: Whether in training mode
        scale_by_keep: Whether to scale by keep probability
        
    Returns:
        Tensor with dropped paths
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = jax.random.bernoulli(rng, keep_prob, shape).astype(x.dtype)
    
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    
    return x * random_tensor


def softmax_with_mask(logits: jnp.ndarray, 
                      mask: Optional[jnp.ndarray] = None,
                      axis: int = -1) -> jnp.ndarray:
    """
    Compute softmax with optional masking.
    
    Args:
        logits: Input logits
        mask: Boolean mask where True indicates valid positions
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax probabilities with masked positions set to 0
    """
    if mask is None:
        return jax.nn.softmax(logits, axis=axis)
    
    # Apply mask by setting invalid positions to large negative value
    large_neg = jnp.finfo(logits.dtype).min
    masked_logits = jnp.where(mask, logits, large_neg)
    
    return jax.nn.softmax(masked_logits, axis=axis)
