"""
JAX/Flax training script for QM9 molecular property prediction using PlatonicTransformer.

This script supports multi-GPU training using JAX's pmap for data parallelism across devices.
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.jax_utils import replicate, unreplicate
from tqdm import tqdm
import ml_collections
import wandb

# JAX-native QM9 dataset (no PyTorch dependencies)
from platonic_transformers.datasets.qm9_jax import (
    QM9DatasetJax,
    DataLoaderJax,
    Batch
)

from platonic_transformers.datasets.k_hot_encoding import KHOT_EMBEDDINGS
from platonic_transformers.models.platoformer_jax import (
    PlatonicTransformer,
    PLATONIC_GROUPS,
    segment_sum,
)
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config
)


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    shift: jnp.ndarray
    scale: jnp.ndarray
    avg_num_nodes: float
    rng: jax.random.PRNGKey


def create_model(config: ml_collections.ConfigDict) -> PlatonicTransformer:
    """Create the PlatonicTransformer model from config."""
    
    solid_name = config.model.solid_name.lower()
    if solid_name not in PLATONIC_GROUPS:
        raise ValueError(f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}")
    
    # Determine input dimension
    input_dim = 97 if config.dataset.use_k_hot_encoding else 11
    
    return PlatonicTransformer(
        input_dim=input_dim,
        input_dim_vec=0,
        hidden_dim=config.model.hidden_dim,
        output_dim=1,  # Single target property
        output_dim_vec=0,
        nhead=config.model.num_heads,
        num_layers=config.model.num_layers,
        solid_name=solid_name,
        spatial_dim=config.model.spatial_dim,
        dense_mode=config.model.dense_mode,
        scalar_task_level=config.model.scalar_task_level,
        vector_task_level=config.model.vector_task_level,
        ffn_readout=config.model.ffn_readout,
        mean_aggregation=config.model.mean_aggregation,
        dropout=config.model.dropout,
        drop_path_rate=config.model.drop_path_rate,
        layer_scale_init_value=config.model.layer_scale_init_value,
        attention=config.model.attention,
        ffn_dim_factor=config.model.ffn_dim_factor,
        rope_sigma=config.model.rope_sigma,
        ape_sigma=config.model.ape_sigma,
        learned_freqs=config.model.learned_freqs,
        freq_init=config.model.freq_init,
        use_key=config.model.use_key,
    )


def create_khot_embedding_table() -> jnp.ndarray:
    """Create k-hot embedding lookup table for QM9 atoms."""
    # QM9 atoms: H(1), C(6), N(7), O(8), F(9)
    embedding_dim = len(next(iter(KHOT_EMBEDDINGS.values())))
    embedding_table = np.zeros((5, embedding_dim), dtype=np.float32)
    qm9_to_atomic = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
    
    for qm9_idx, atomic_num in qm9_to_atomic.items():
        if atomic_num in KHOT_EMBEDDINGS:
            embedding_table[qm9_idx] = np.array(KHOT_EMBEDDINGS[atomic_num])
    
    return jnp.array(embedding_table)


def apply_khot_encoding(x: jnp.ndarray, khot_table: jnp.ndarray) -> jnp.ndarray:
    """Apply k-hot encoding to node features."""
    # x has one-hot atom type in first 5 columns
    atom_indices = jnp.argmax(x[:, :5], axis=-1)
    embeddings = khot_table[atom_indices]
    # Concatenate with remaining features
    return jnp.concatenate([embeddings, x[:, -5:]], axis=-1)


def batch_to_jax(batch: Batch, use_khot: bool, khot_table: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """Convert Batch (numpy arrays) to JAX arrays."""
    x = jnp.array(batch.x)
    pos = jnp.array(batch.pos)
    batch_idx = jnp.array(batch.batch)
    y = jnp.array(batch.y)
    
    if use_khot and khot_table is not None:
        x = apply_khot_encoding(x, khot_table)
    
    return {
        'x': x,
        'pos': pos,
        'batch': batch_idx,
        'y': y,
        'num_graphs': batch.num_graphs,  # Pass as concrete value before tracing
    }


def center_positions(pos: jnp.ndarray, batch: jnp.ndarray, num_graphs: int) -> jnp.ndarray:
    """Center positions per graph by subtracting mean."""
    # Compute mean position per graph
    pos_sum = segment_sum(pos, batch, num_graphs)
    counts = segment_sum(jnp.ones((pos.shape[0], 1)), batch, num_graphs)
    pos_mean = pos_sum / jnp.maximum(counts, 1.0)
    
    # Center positions
    return pos - pos_mean[batch]


def random_rotation_matrix(rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate a random SO(3) rotation matrix."""
    # QR decomposition method for uniform random rotations
    rngs = jax.random.split(rng, 3)
    A = jax.random.normal(rngs[0], (3, 3))
    Q, R = jnp.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    signs = jnp.sign(jnp.diag(R))
    Q = Q * signs
    if jnp.linalg.det(Q) < 0:
        Q = Q.at[:, 0].multiply(-1)
    return Q


def apply_rotation_augmentation(
    pos: jnp.ndarray, 
    batch: jnp.ndarray, 
    rng: jax.random.PRNGKey,
    num_graphs: int
) -> jnp.ndarray:
    """Apply random rotation augmentation per graph."""
    # Generate rotation matrices for each graph
    rngs = jax.random.split(rng, num_graphs)
    rotations = jax.vmap(random_rotation_matrix)(rngs)  # [num_graphs, 3, 3]
    
    # Apply rotation to each node
    rot_per_node = rotations[batch]  # [N, 3, 3]
    return jnp.einsum('nij,nj->ni', rot_per_node, pos)


def compute_loss(
    params: Any,
    state: TrainState,
    batch_data: Dict[str, jnp.ndarray],
    model: PlatonicTransformer,
    train: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
    train_augm: bool = False,
    num_graphs: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute MAE loss for a batch."""
    x = batch_data['x']
    pos = batch_data['pos']
    batch_idx = batch_data['batch']
    y = batch_data['y']
    
    # Center positions
    pos = center_positions(pos, batch_idx, num_graphs)
    
    # Apply rotation augmentation if training
    if train and train_augm and rng is not None:
        pos = apply_rotation_augmentation(pos, batch_idx, rng, num_graphs)
    
    # Forward pass
    pred, _ = model.apply(
        {'params': params},
        x=x,
        pos=pos,
        batch=batch_idx,
        deterministic=not train,
        avg_num_nodes=state.avg_num_nodes,
    )
    pred = pred.squeeze(-1)  # [num_graphs]
    
    # Normalize target
    y_normalized = (y - state.shift) / state.scale
    
    # MAE loss
    loss = jnp.mean(jnp.abs(pred - y_normalized))
    
    # Denormalized MAE for logging
    mae = jnp.mean(jnp.abs(pred * state.scale + state.shift - y))
    
    return loss, mae


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2, 4, 5))
def train_step(
    state: TrainState,
    batch_data: Dict[str, jnp.ndarray],
    model: PlatonicTransformer,
    rng: jax.random.PRNGKey,
    train_augm: bool,
    num_graphs: int
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """Parallel training step."""
    
    def loss_fn(params):
        return compute_loss(params, state, batch_data, model, train=True, rng=rng, train_augm=train_augm, num_graphs=num_graphs)
    
    (loss, mae), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    mae = jax.lax.pmean(mae, axis_name='batch')
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, mae


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2, 3))
def eval_step(
    state: TrainState,
    batch_data: Dict[str, jnp.ndarray],
    model: PlatonicTransformer,
    num_graphs: int
) -> jnp.ndarray:
    """Parallel evaluation step."""
    _, mae = compute_loss(state.params, state, batch_data, model, train=False, num_graphs=num_graphs)
    mae = jax.lax.pmean(mae, axis_name='batch')
    return mae


def create_train_state(
    rng: jax.random.PRNGKey,
    model: PlatonicTransformer,
    config: ml_collections.ConfigDict,
    shift: float,
    scale: float,
    avg_num_nodes: float
) -> TrainState:
    """Initialize training state."""
    
    # Dummy inputs for initialization
    dummy_x = jnp.zeros((10, 97 if config.dataset.use_k_hot_encoding else 11))
    dummy_pos = jnp.zeros((10, 3))
    dummy_batch = jnp.zeros(10, dtype=jnp.int32)
    
    params = model.init(
        rng,
        x=dummy_x,
        pos=dummy_pos,
        batch=dummy_batch,
        deterministic=True,
        avg_num_nodes=avg_num_nodes
    )['params']
    
    # Create optimizer with warmup cosine schedule
    total_steps = config.training.epochs * 1000  # Approximate
    warmup_steps = config.scheduler.warmup_epochs * 1000
    
    if config.scheduler.use_cosine:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=config.optimizer.lr * 0.01
        )
    else:
        schedule = config.optimizer.lr
    
    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.gradient_clip_val),
        optax.adamw(learning_rate=schedule, weight_decay=config.optimizer.weight_decay)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        shift=jnp.array(shift),
        scale=jnp.array(scale),
        avg_num_nodes=avg_num_nodes,
        rng=rng
    )


def compute_dataset_statistics(dataloader: DataLoaderJax, target_idx: int) -> Tuple[float, float, float]:
    """Compute mean, std, and avg_num_nodes from training data."""
    ys = []
    total_nodes = 0
    total_graphs = 0
    
    for batch in tqdm(dataloader, desc="Computing statistics"):
        ys.append(batch.y)
        total_nodes += batch.num_nodes
        total_graphs += batch.num_graphs
    
    ys = np.concatenate(ys)
    shift = float(np.mean(ys))
    scale = float(np.std(ys))
    avg_num_nodes = float(total_nodes / total_graphs)
    
    return shift, scale, avg_num_nodes


def load_data(config: ml_collections.ConfigDict) -> Tuple[DataLoaderJax, DataLoaderJax, DataLoaderJax, int]:
    """Load QM9 dataset using JAX-native dataset class."""
    # Target selection
    all_targets = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
        'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
    ]
    target_map = {name: i for i, name in enumerate(all_targets)}
    
    target_name = config.dataset.target
    atom_version = f"{target_name}_atom"
    if atom_version in target_map:
        print(f"Using atomization energy target: {atom_version}")
        target_name = atom_version
    
    target_idx = target_map[target_name]
    
    # Load datasets with JAX-native dataset class
    print ("[DEBUG] DATADIR:",config.dataset.data_dir)
    datasets = {
        split: QM9DatasetJax(
            root=config.dataset.data_dir,
            target=target_name,
            split=split
        )
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoaderJax(
            split_dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == 'train'),
            drop_last=(split == 'train')  # Important for multi-GPU
        )
        for split, split_dataset in datasets.items()
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test'], target_idx


def shard_batch(batch_data: Dict[str, jnp.ndarray], num_devices: int) -> Dict[str, jnp.ndarray]:
    """Shard batch data across devices."""
    # Note: For graph data with variable sizes, we need to handle this carefully
    # Here we assume batches can be evenly divided
    def shard_array(arr):
        # Pad if necessary
        batch_size = arr.shape[0]
        remainder = batch_size % num_devices
        if remainder != 0:
            pad_size = num_devices - remainder
            pad_shape = (pad_size,) + arr.shape[1:]
            arr = jnp.concatenate([arr, jnp.zeros(pad_shape, dtype=arr.dtype)], axis=0)
        return arr.reshape(num_devices, -1, *arr.shape[1:])
    
    # num_graphs is handled separately as static arg, skip it here
    return {k: shard_array(v) for k, v in batch_data.items() if k != 'num_graphs'}


def train_epoch(
    state: TrainState,
    train_loader: DataLoaderJax,
    model: PlatonicTransformer,
    use_khot: bool,
    khot_table: jnp.ndarray,
    train_augm: bool,
    num_devices: int
) -> Tuple[TrainState, float, float]:
    """Train for one epoch."""
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Convert to JAX
        batch_data = batch_to_jax(batch, use_khot, khot_table)
        
        # Extract num_graphs before sharding (it's a static value)
        num_graphs = batch_data['num_graphs']
        
        # Shard across devices
        batch_data = shard_batch(batch_data, num_devices)
        
        # Get RNG for this step - unreplicate first since state.rng is replicated
        rng_single = unreplicate(state.rng)
        rng_keys = jax.random.split(rng_single, num_devices + 1)
        # Use first key to update state for next iteration, rest for devices
        next_rng = rng_keys[0]
        rng = rng_keys[1:]  # Shape: (num_devices, 2)
        
        # Training step
        state, loss, mae = train_step(state, batch_data, model, rng, train_augm, num_graphs)
        
        # Update state's rng for next batch
        state = state.replace(rng=replicate(next_rng))
        
        total_loss += float(unreplicate(loss))
        total_mae += float(unreplicate(mae))
        num_batches += 1
    
    return state, total_loss / num_batches, total_mae / num_batches


def evaluate(
    state: TrainState,
    dataloader: DataLoaderJax,
    model: PlatonicTransformer,
    use_khot: bool,
    khot_table: jnp.ndarray,
    num_devices: int,
    desc: str = "Evaluating"
) -> float:
    """Evaluate on a dataset."""
    total_mae = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        batch_data = batch_to_jax(batch, use_khot, khot_table)
        num_graphs = batch_data['num_graphs']
        batch_data = shard_batch(batch_data, num_devices)
        
        mae = eval_step(state, batch_data, model, num_graphs)
        total_mae += float(unreplicate(mae))
        num_batches += 1
    
    return total_mae / num_batches


def main(config: ml_collections.ConfigDict) -> None:
    """Main training function."""
    print_config(config, "QM9 JAX Training Configuration")
    
    # Set random seed
    np.random.seed(config.seed)
    rng = jax.random.PRNGKey(config.seed)
    
    # Setup devices
    num_devices = jax.local_device_count()
    print(f"Number of JAX devices: {num_devices}")
    print(f"Device names: {[d.device_kind for d in jax.devices()]}")
    
    # Load data
    train_loader, val_loader, test_loader, target_idx = load_data(config)
    
    # Create k-hot table
    khot_table = create_khot_embedding_table() if config.dataset.use_k_hot_encoding else None
    
    # Compute dataset statistics
    data_dir = os.path.expanduser(config.dataset.data_dir)
    stats_file = os.path.join(data_dir, f"stats_{config.dataset.target}.npz")
    if os.path.exists(stats_file):
        print(f"Loading statistics from {stats_file}")
        stats = np.load(stats_file)
        shift, scale, avg_num_nodes = float(stats['shift']), float(stats['scale']), float(stats['avg_num_nodes'])
    else:
        print("Computing dataset statistics...")
        shift, scale, avg_num_nodes = compute_dataset_statistics(train_loader, target_idx)
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        np.savez(stats_file, shift=shift, scale=scale, avg_num_nodes=avg_num_nodes)
    
    print(f"Target statistics - Mean: {shift:.4f}, Std: {scale:.4f}, Avg nodes: {avg_num_nodes:.2f}")
    
    # Create model and state
    model = create_model(config)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, config, shift, scale, avg_num_nodes)
    
    # Replicate state across devices
    state = replicate(state)
    
    # Initialize wandb logging
    if config.logging.enabled:
        wandb.init(
            project=config.logging.project_name + "-JAX",
            config=config.to_dict(),
            mode="offline"  # TODO: Remove for production
        )
    
    # Training loop
    best_val_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(config.training.epochs):
        # Train
        state, train_loss, train_mae = train_epoch(
            state, train_loader, model,
            config.dataset.use_k_hot_encoding, khot_table,
            config.training.train_augm, num_devices
        )
        
        # Validate
        val_mae = evaluate(
            state, val_loader, model,
            config.dataset.use_k_hot_encoding, khot_table,
            num_devices, "Validating"
        )
        
        # Logging
        print(f"Epoch {epoch+1}/{config.training.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        
        if config.logging.enabled:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
            })
        
        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            
            # Save checkpoint
            ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints_jax")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir,
                unreplicate(state),
                epoch + 1,
                prefix='best_',
                keep=1
            )
            print(f"  -> New best model saved! Val MAE: {val_mae:.4f}")
    
    # Final test evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints_jax")
    state = checkpoints.restore_checkpoint(ckpt_dir, unreplicate(state), prefix='best_')
    state = replicate(state)
    
    test_mae = evaluate(
        state, test_loader, model,
        config.dataset.use_k_hot_encoding, khot_table,
        num_devices, "Testing"
    )
    
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    if config.logging.enabled:
        wandb.log({'test_mae': test_mae, 'best_epoch': best_epoch, 'best_val_mae': best_val_mae})
        wandb.finish()


if __name__ == "__main__":
    # Parse arguments
    parser = get_arg_parser(default_config_path="configs/qm9_regr.yaml")
    args, unknown_args = parser.parse_known_args()
    
    # Load configuration
    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args
    )
    
    # Run training
    main(config)