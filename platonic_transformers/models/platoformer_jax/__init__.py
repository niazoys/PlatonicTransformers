"""
Platonic Transformer implementation in JAX/Flax.

This module provides a JAX implementation of the Platonic Transformer,
a Transformer architecture equivariant to the symmetries of Platonic solids.
"""

from platonic_transformers.models.platoformer_jax.groups import (
    PlatonicSolidGroup,
    PLATONIC_GROUPS,
    get_group,
)

from platonic_transformers.models.platoformer_jax.linear import (
    PlatonicLinear,
    EquivariantLayerNorm,
)

from platonic_transformers.models.platoformer_jax.rope import PlatonicRoPE
from platonic_transformers.models.platoformer_jax.ape import APE, PlatonicAPE
from platonic_transformers.models.platoformer_jax.conv import PlatonicConv
from platonic_transformers.models.platoformer_jax.block import PlatonicBlock, DropPath

from platonic_transformers.models.platoformer_jax.platoformer import (
    PlatonicTransformer,
    create_platoformer,
)

from platonic_transformers.models.platoformer_jax.io import (
    lift,
    lift_scalars,
    lift_vectors,
    pool,
    to_scalars_vectors,
    to_dense_and_mask,
    readout_scalars,
    readout_vectors,
)

from platonic_transformers.models.platoformer_jax.utils import (
    scatter_add,
    scatter_mean,
    segment_sum,
    segment_mean,
    drop_path,
    softmax_with_mask,
)


__all__ = [
    # Groups
    'PlatonicSolidGroup',
    'PLATONIC_GROUPS',
    'get_group',
    
    # Layers
    'PlatonicLinear',
    'EquivariantLayerNorm',
    'PlatonicRoPE',
    'APE',
    'PlatonicAPE',
    'PlatonicConv',
    'PlatonicBlock',
    'DropPath',
    
    # Model
    'PlatonicTransformer',
    'create_platoformer',
    
    # I/O utilities
    'lift',
    'lift_scalars',
    'lift_vectors',
    'pool',
    'to_scalars_vectors',
    'to_dense_and_mask',
    'readout_scalars',
    'readout_vectors',
    
    # Utilities
    'scatter_add',
    'scatter_mean',
    'segment_sum',
    'segment_mean',
    'drop_path',
    'softmax_with_mask',
]
