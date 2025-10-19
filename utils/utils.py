import math
import numbers
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Callbacks are now in utils.callbacks
from utils.callbacks import (
    MemoryMonitorCallback,
    NaNDetectorCallback,
    StopOnPersistentDivergence,
    TimerCallback,
    TrainingTimerCallback,
)


def run_gc() -> None:
    """Free Python and CUDA memory caches."""

    gc.collect()
    torch.cuda.empty_cache()


def format_batch_for_esen(batch: Any) -> Any:
    """Convert an OMol batch into the structure expected by the eSEN model."""

    class AtomicDataObj:
        """Dictionary-like container with attribute access support."""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._dict: Dict[Any, Any] = kwargs

        def __getitem__(self, key: Any) -> Any:
            if key in self._dict:
                return self._dict[key]
            if hasattr(self, str(key)):
                return getattr(self, str(key))
            raise KeyError(f"Key '{key}' not found")

        def __setitem__(self, key: Any, value: Any) -> None:
            self._dict[key] = value
            if isinstance(key, str):
                setattr(self, key, value)
            elif isinstance(key, int):
                return
            else:
                try:
                    setattr(self, str(key), value)
                except (TypeError, ValueError):
                    pass

        def get(self, key: Any, default: Any = None) -> Any:
            return self._dict.get(key, default)

        def keys(self) -> Iterable[Any]:
            return self._dict.keys()

        def __contains__(self, key: Any) -> bool:
            return key in self._dict

        def __len__(self) -> int:
            if hasattr(self, "natoms"):
                return len(self.natoms)
            return 0

    systems: List[AtomicDataObj] = []
    natoms = batch.num_atoms.tolist()

    start_idx = 0
    for i, num_atoms in enumerate(natoms):
        end_idx = start_idx + num_atoms
        system = AtomicDataObj(
            pos=batch.pos[start_idx:end_idx],
            atomic_numbers=batch.atomic_numbers[start_idx:end_idx],
            cell=batch.cell[i : i + 1]
            if hasattr(batch, "cell")
            else torch.eye(3, device=batch.pos.device).unsqueeze(0) * 20.0,
            pbc=torch.zeros(1, 3, dtype=torch.bool, device=batch.pos.device),
            natoms=torch.tensor([num_atoms], device=batch.pos.device),
        )
        systems.append(system)
        start_idx = end_idx

    data_dict = AtomicDataObj(
        pos=batch.pos,
        atomic_numbers=batch.atomic_numbers,
        batch=batch.batch,
        natoms=torch.tensor(natoms, device=batch.pos.device),
        charge=
        batch.charge
        if hasattr(batch, "charge")
        else torch.zeros(len(natoms), device=batch.pos.device),
        spin=
        batch.spin
        if hasattr(batch, "spin")
        else torch.zeros(len(natoms), device=batch.pos.device),
        dataset=["default"] * len(natoms),
        systems=systems,
    )

    if hasattr(batch, "cell"):
        data_dict.cell = batch.cell
    else:
        data_dict.cell = (
            torch.eye(3, device=batch.pos.device)
            .unsqueeze(0)
            .repeat(len(natoms), 1, 1)
            * 20.0
        )

    if hasattr(batch, "pbc"):
        data_dict.pbc = batch.pbc
    else:
        data_dict.pbc = torch.zeros(len(natoms), 3, dtype=torch.bool, device=batch.pos.device)

    for i, system in enumerate(systems):
        data_dict[i] = system

    return data_dict


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with a warmup phase."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int) -> None:
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) / (self.warmup + 1e-6)
        return float(lr_factor)


class RandomRotateWithNormals(BaseTransform):
    r"""Rotate node positions around a specific axis by a random angle."""

    def __init__(self, degrees: Union[Tuple[float, float], float], axis: int = 0) -> None:
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data: Data) -> Data:
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformationWithNormals(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degrees}, axis={self.axis})"


class LinearTransformationWithNormals(BaseTransform):
    r"""Apply a fixed linear transformation to positions and normals."""

    def __init__(self, matrix: torch.Tensor) -> None:
        assert matrix.dim() == 2, "Transformation matrix should be two-dimensional."
        assert matrix.size(0) == matrix.size(1), "Transformation matrix should be square."
        self.matrix = matrix.t()

    def __call__(self, data: Data) -> Data:
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
        norm = data.x.view(-1, 1) if data.x.dim() == 1 else data.x

        assert pos.size(-1) == self.matrix.size(-2), "Incompatible position and matrix shapes."
        assert norm.size(-1) == self.matrix.size(-2), "Incompatible normal and matrix shapes."

        mat = self.matrix.to(pos.dtype).to(pos.device)
        data.pos = torch.matmul(pos, mat)
        data.x = torch.matmul(norm, mat)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix.tolist()})"


class SamplePoints(BaseTransform):
    """Uniformly sample points on mesh faces based on face area."""

    def __init__(self, num: int, remove_faces: bool = True, include_normals: bool = False) -> None:
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None

        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        area = torch.linalg.cross(vec1, vec2, dim=1)
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            normals = torch.linalg.cross(vec1, vec2, dim=1)
            data.normal = torch.nn.functional.normalize(normals, p=2, dim=1)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data


class NormalizeCoord(BaseTransform):
    """Normalize point-cloud coordinates by centering and scaling."""

    def __call__(self, data: Data) -> Data:
        centroid = torch.mean(data.pos, dim=0)
        data.pos = data.pos - centroid

        distances = torch.sqrt(torch.sum(data.pos ** 2, dim=1))
        scale = torch.max(distances)
        data.pos = data.pos / scale

        return data


class RandomJitter(BaseTransform):
    """Randomly jitter points by adding normal noise."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05, relative: bool = False) -> None:
        self.sigma = sigma
        self.clip = clip
        self.relative = relative

    def __call__(self, data: Data) -> Data:
        if self.relative:
            scale = torch.std(torch.norm(data.pos, dim=1))
            sigma = self.sigma * scale
            clip = self.clip * scale
        else:
            sigma = self.sigma
            clip = self.clip
        noise = torch.clamp(sigma * torch.randn_like(data.pos), min=-clip, max=clip)
        data.pos = data.pos + noise
        return data


class RandomShift(BaseTransform):
    """Randomly shift the point cloud."""

    def __init__(self, shift_range: float = 0.1) -> None:
        self.shift_range = shift_range

    def __call__(self, data: Data) -> Data:
        shift = torch.rand(3, device=data.pos.device) * 2 * self.shift_range - self.shift_range
        data.pos = data.pos + shift
        return data


class RandomRotatePerturbation(BaseTransform):
    """Apply small random rotations around all axes."""

    def __init__(self, angle_sigma: float = 0.06, angle_clip: float = 0.18) -> None:
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data: Data) -> Data:
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3, device=data.pos.device),
            min=-self.angle_clip,
            max=self.angle_clip,
        )

        cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

        dtype = data.pos.dtype
        Rx = torch.tensor(
            [[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]],
            device=data.pos.device,
            dtype=dtype,
        )
        Ry = torch.tensor(
            [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]],
            device=data.pos.device,
            dtype=dtype,
        )
        Rz = torch.tensor(
            [[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]],
            device=data.pos.device,
            dtype=dtype,
        )

        R = torch.mm(torch.mm(Rz, Ry), Rx)
        data.pos = torch.mm(data.pos, R.t())
        if hasattr(data, "normal"):
            data.normal = torch.mm(data.normal, R.t())
        return data


def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode a tensor of class indices."""

    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Compute the mean of ``src`` values grouped by indices along ``dim``."""

    out_shape = [dim_size] + list(src.shape[1:])
    out_sum = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    out_sum.scatter_add_(dim, index_expanded, src)

    ones = torch.ones_like(src)
    out_count = torch.zeros(out_shape, dtype=torch.float, device=src.device)
    out_count.scatter_add_(dim, index_expanded, ones)
    out_count[out_count == 0] = 1

    return out_sum / out_count


def fully_connected_edge_index(batch_idx: torch.Tensor) -> torch.Tensor:
    """Construct fully connected edge indices within each batch element."""

    edge_indices = []
    for batch_num in torch.unique(batch_idx):
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing="ij")
        edge_indices.append(torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0))
    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index


def subtract_mean(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Subtract per-graph means from batched coordinates."""

    means = scatter_mean(src=pos, index=batch, dim=0, dim_size=batch.max().item() + 1)
    return pos - means[batch]


class RandomSOd(torch.nn.Module):
    r"""Generate random rotation matrices in :math:`\mathrm{SO}(2)` or :math:`\mathrm{SO}(3)`."""

    def __init__(self, d: int) -> None:
        super().__init__()
        assert d in [2, 3], "d must be 2 or 3."
        self.d = d

    def forward(self, n: Optional[int] = None) -> torch.Tensor:
        if self.d == 2:
            return self._generate_2d(n)
        return self._generate_3d(n)

    def _generate_2d(self, n: Optional[int]) -> torch.Tensor:
        theta = torch.rand(n) * 2 * torch.pi if n else torch.rand(1) * 2 * torch.pi
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
        if n:
            return rotation_matrix.view(n, 2, 2)
        return rotation_matrix.view(2, 2)

    def _generate_3d(self, n: Optional[int]) -> torch.Tensor:
        q = torch.randn(n, 4) if n else torch.randn(4)
        q = q / torch.norm(q, dim=-1, keepdim=True)
        q0, q1, q2, q3 = q.unbind(-1)
        rotation_matrix = torch.stack(
            [
                1 - 2 * (q2 ** 2 + q3 ** 2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1 ** 2 + q3 ** 2),
                2 * (q2 * q3 - q0 * q1),
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1 ** 2 + q2 ** 2),
            ],
            dim=-1,
        )
        if n:
            return rotation_matrix.view(n, 3, 3)
        return rotation_matrix.view(3, 3)


class RandomSO2AroundAxis(torch.nn.Module):
    def __init__(self, axis: int = 2, degrees: Union[float, Tuple[float, float]] = 15) -> None:
        super().__init__()
        assert axis in [0, 1, 2], "axis must be 0 (X), 1 (Y), or 2 (Z)"
        self.axis = axis

        if isinstance(degrees, (float, int)):
            self.degrees = (-abs(float(degrees)), abs(float(degrees)))
        elif isinstance(degrees, (tuple, list)):
            assert len(degrees) == 2, "degrees tuple must have length 2"
            self.degrees = tuple(map(float, degrees))
        else:
            raise ValueError("degrees must be a number or a tuple")

    def forward(self, n: Optional[int] = None) -> torch.Tensor:
        min_deg, max_deg = self.degrees
        angles = torch.rand(n if n else 1) * (max_deg - min_deg) + min_deg
        theta = angles * torch.pi / 180.0
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        if self.axis == 0:
            rotation_matrix = torch.stack(
                [
                    torch.ones_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    cos_theta,
                    sin_theta,
                    torch.zeros_like(cos_theta),
                    -sin_theta,
                    cos_theta,
                ],
                dim=-1,
            )
        elif self.axis == 1:
            rotation_matrix = torch.stack(
                [
                    cos_theta,
                    torch.zeros_like(cos_theta),
                    -sin_theta,
                    torch.zeros_like(cos_theta),
                    torch.ones_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    sin_theta,
                    torch.zeros_like(cos_theta),
                    cos_theta,
                ],
                dim=-1,
            )
        else:
            rotation_matrix = torch.stack(
                [
                    cos_theta,
                    sin_theta,
                    torch.zeros_like(cos_theta),
                    -sin_theta,
                    cos_theta,
                    torch.zeros_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    torch.zeros_like(cos_theta),
                    torch.ones_like(cos_theta),
                ],
                dim=-1,
            )

        if n:
            return rotation_matrix.view(n, 3, 3)
        return rotation_matrix.view(3, 3)

