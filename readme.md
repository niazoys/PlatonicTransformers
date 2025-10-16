# Platonic Transformer

Welcome to the Platonic Transformer project, where geometric group theory meets modern attention architectures 🌟. This repository contains research code for Platonic-Group-equivariant Transformers that operate on scalar and vector features operating over the frames defined by Platonic solid symmetry groups. We combine group convolutions and (linear) attention to deliver accurate, symmetry-aware models that are as fast as `torch.nn.TransformerEncoder`.

## Why You May Love This Project
- **Group-equivariant attention** that respects the symmetry group of the chosen Platonic solid (tetrahedron, octahedron, icosahedron 🧊).
- **Unified scalar/vector processing** with shared Platonic blocks for graph- and node-level predictions.
- **Supports multiple benchmarks** including CIFAR-10, ImageNet-1k, QM9 regression, ModelNet40, ShapeNet Cars, Protein folding, and Open Molecule Learning.
- **WandB integration** for centralized experiment tracking with teammates.

## Repository Tour
```
.
├── datasets/                # Dataset wrappers and loaders for supported benchmarks
├── main_*.py                # Entry points for training on specific datasets (CIFAR-10, QM9, etc.)
├── models/
│   └── platoformer/         # Platonic Transformer building blocks
│       ├── block.py         # Core PlatonicBlock attention + feedforward module
│       ├── conv.py          # Group convolution utilities
│       ├── groups.py        # Symmetry group definitions for Platonic solids
│       ├── io.py            # Lifting, pooling, dense/sparse utilities
│       ├── linear.py        # Equivariant linear projections
│       └── platoformer.py   # Full PlatonicTransformer module
├── utils.py                # Shared training utilities (logging, metrics, augmentation)
├── setup.sh                # Environment bootstrapper
└── readme.md               # You are here
```

## Getting Started 🚀

1. **Clone the repository** and install system dependencies if needed.
2. **Create the environment:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```
4. **Authenticate with Weights & Biases** :
   ```bash
   wandb login
   ```
> For most tasks, our models were trained on a single GPU (eg: A100/H200). We recommend having at least 32GiB VRAM.

## Using Platonic Transformer

The Platonic Transformer layer is a stand-in replacement for `torch.nn.TransformerEncoder`. We barely introduce any overhead, making the Platonic Transformer blazingly fast during training and inference.

```python
from platoformer import PlatonicTransformer

class MyCoolNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # the old you:
        self.layer = torch.nn.TransformerEncoder(...)

        # the new, cooler you:
        self.layer = PlatonicTransformer(...)

    def forward(self, node_ftrs, node_pos):
        x = self.layer(node_ftrs, node_pos)
```

## Training Workflows 🏋️‍♀️
Each `main_*.py` script exposes dataset-specific defaults. Pass `--help` for full CLI options, or keep things simple with the out-of-the-box settings:

Most scripts share common flags such as:
- `--solid_name {tetrahedron, octahedron, icosahedron}` to pick the symmetry group.
- `--hidden_dim`, `--layers`, `--num_heads` for model capacity.
- `--rope_sigma` / `--ape_sigma` to toggle rotational and absolute positional embeddings.
- `--conditioning_dim` when injecting diffusion or guidance signals.

Tip: start with smaller `--hidden-dim` (e.g., 64) and fewer layers to validate pipelines quickly 😊.

## Platonic Transformer Anatomy 🧠
The heart of the project sits in `models/platoformer/platoformer.py`:
- **Lifting:** `lift` maps scalar and vector node features to group-aligned channels.
- **Attention Blocks:** `PlatonicBlock` layers (stacked in `self.layers`) combine group-aware attention and equivariant MLPs with optional AdaLayerNorm-style conditioning.
- **Positional Encoding:** Choose between rotational positional encodings (RoPE) and absolute encodings (APE) tuned per solid.
- **Readout:** Separate scalar/vector readouts followed by pooling yield graph-level or node-level predictions, configurable via `scalar_task_level` and `vector_task_level`.


