"""ASE calculator interface for Platonic Transformer force fields.

Wraps :class:`platonic_transformers.models.platoformer.force_field.PlatonicForceField`
(the OMol25 energy/force model trained by ``mains/main_omol.py``) as an
``ase.calculators.calculator.Calculator`` so it can drive ASE relaxations, MD and
speed benchmarks.

Units follow OMol25 / ASE conventions: positions in Angstrom, energies in eV, forces
in eV/Angstrom. The network itself predicts in a normalised space (per-element
reference energies subtracted, then divided by the training-set RMSD); the
calculator undoes both steps, so ``atoms.get_potential_energy()`` is the total
energy in eV.

Example::

    from ase.build import molecule
    from platonic_transformers.calculator import PlatonicCalculator

    calc = PlatonicCalculator.from_checkpoint("path/to/last.ckpt", device="cuda")
    atoms = molecule("H2O")
    atoms.info["charge"], atoms.info["spin"] = 0, 1
    atoms.calc = calc
    print(atoms.get_potential_energy(), atoms.get_forces())

``ase`` is an optional dependency: ``pip install "platonic_transformers[ase]"``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError as _ase_import_error:  # pragma: nocover
    raise ImportError(
        "platonic_transformers.calculator requires the `ase` package. Install it with "
        "`pip install ase` or `pip install \"platonic_transformers[ase]\"`."
    ) from _ase_import_error

from platonic_transformers.models.platoformer.force_field import (
    PlatonicForceField,
    build_force_field,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["PlatonicCalculator", "load_force_field_checkpoint"]


def _as_dict(cfg: Any) -> dict:
    """Return a plain dict for a dict / ml_collections.ConfigDict / namespace."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(vars(cfg))


def _load_element_refs(path: Union[str, os.PathLike]) -> torch.Tensor:
    """Read the ``omol_elem_refs`` table (eV per element, indexed by Z) from YAML."""
    import yaml

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = _REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    refs = data.get("omol_elem_refs", data) if isinstance(data, dict) else data
    return torch.as_tensor(refs, dtype=torch.float64)


def load_force_field_checkpoint(
    ckpt_path: Union[str, os.PathLike],
    config: Optional[Union[str, os.PathLike, Mapping[str, Any]]] = None,
    use_ema: bool = True,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> tuple[PlatonicForceField, dict]:
    """Rebuild a :class:`PlatonicForceField` from a ``mains/main_omol.py`` checkpoint.

    The Lightning ``.ckpt`` stores the full training config under
    ``hyper_parameters["config"]``; the ``net.*`` entries of its ``state_dict`` are the
    force-field weights, and ``scale`` / ``shift`` / ``element_refs`` are the energy
    normalisation buffers of ``OMolModel``.

    Args:
        ckpt_path: Path to a Lightning checkpoint (``last.ckpt``, ``best-*.ckpt``).
        config: Optional override for the training config -- a YAML path (e.g.
            ``configs/omol.yaml``) or a mapping with a ``model`` section. Needed only
            for checkpoints that do not carry ``hyper_parameters``.
        use_ema: Load the EMA shadow weights saved by ``EMACallback`` when present
            (this is what validation/testing in ``main_omol.py`` evaluates).
        map_location: Passed to :func:`torch.load`.
        strict: Passed to :meth:`torch.nn.Module.load_state_dict`.

    Returns:
        ``(model, meta)`` where ``meta`` has keys ``scale`` (float), ``shift`` (float),
        ``element_refs`` (``torch.Tensor`` or ``None``), ``config`` (dict) and
        ``ema_loaded`` (bool).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    if config is None:
        hparams = ckpt.get("hyper_parameters", {}) or {}
        cfg = _as_dict(hparams.get("config", hparams))
        if "model" not in cfg:
            raise ValueError(
                f"{ckpt_path} has no `hyper_parameters.config.model`; pass `config=`"
                " (a YAML path such as configs/omol.yaml or a dict with a `model` key)."
            )
    elif isinstance(config, (str, os.PathLike)):
        import yaml

        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = _as_dict(config)

    model = build_force_field(cfg["model"])

    # Force-field weights live under the `net.` prefix of the LightningModule.
    net_state = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    if not net_state:  # a bare PlatonicForceField state_dict
        net_state = {k: v for k, v in state_dict.items()
                     if k not in ("scale", "shift", "element_refs", "avg_num_nodes")}

    ema_loaded = False
    if use_ema and isinstance(ckpt.get("callbacks"), dict):
        for key, cb_state in ckpt["callbacks"].items():
            if "EMACallback" in str(key) and isinstance(cb_state, dict) and cb_state.get("ema_state"):
                for name, value in cb_state["ema_state"].items():
                    if name.startswith("net."):
                        target = name[len("net."):]
                        net_state[target] = value.to(net_state[target].dtype)
                ema_loaded = True
                break

    model.load_state_dict(net_state, strict=strict)
    model.eval()

    scale = float(state_dict["scale"]) if "scale" in state_dict else 1.0
    shift = float(state_dict["shift"]) if "shift" in state_dict else 0.0
    if "element_refs" in state_dict:
        element_refs = state_dict["element_refs"].to(torch.float64)
    else:
        refs_path = _as_dict(cfg.get("dataset")).get("element_refs_path")
        element_refs = _load_element_refs(refs_path) if refs_path else None

    meta = {"scale": scale, "shift": shift, "element_refs": element_refs,
            "config": cfg, "ema_loaded": ema_loaded}
    return model, meta


class PlatonicCalculator(Calculator):
    """ASE :class:`~ase.calculators.calculator.Calculator` around a Platonic force field.

    Args:
        model: A :class:`PlatonicForceField` (any ``nn.Module`` with the same
            ``forward(data) -> {"energy": (B,), "forces": (N, 3)}`` contract works).
            Use :meth:`from_checkpoint` to build one from a training checkpoint.
        scale: Energy/force normalisation factor used in training (``OMolModel.scale``,
            the de-referenced training-set RMSD). Model outputs are multiplied by it.
        shift: Additive energy shift in eV applied after ``scale`` (``OMolModel.shift``).
        element_refs: Per-element reference energies in eV indexed by atomic number
            (``omol_elem_refs`` in ``configs/constants/element_refs.yaml``). Their sum
            over the atoms is added back to the predicted energy. ``None`` disables it.
        device: ``torch.device`` or string; defaults to CUDA when available.
        dtype: Floating dtype for model weights and inputs (``torch.float32`` default).
        forces: ``"direct"`` uses the model's equivariant force head (how the OMol
            models are trained, and the fastest option); ``"autograd"`` returns
            ``-dE/dR`` of the predicted energy instead, which is exactly conservative.
        default_charge: Total charge used when ``atoms.info["charge"]`` is absent.
        default_spin: Spin multiplicity used when ``atoms.info["spin"]`` is absent.
            OMol25 stores multiplicity (1 = singlet), hence the default of 1.
        **kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.

    Periodic systems are not supported: the OMol force fields are trained on isolated
    molecules and the model attends over all atoms of a structure without minimum-image
    conventions, so ``stress`` is not implemented and any ``Atoms`` with ``pbc`` set
    raises.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        scale: float = 1.0,
        shift: float = 0.0,
        element_refs: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        forces: str = "direct",
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if forces not in ("direct", "autograd"):
            raise ValueError(f"forces must be 'direct' or 'autograd', got {forces!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.force_mode = forces
        self.scale = float(scale)
        self.shift = float(shift)
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        if element_refs is None:
            self.element_refs = None
        else:
            self.element_refs = torch.as_tensor(element_refs, dtype=torch.float64).to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, os.PathLike],
        config: Optional[Union[str, os.PathLike, Mapping[str, Any]]] = None,
        use_ema: bool = True,
        **kwargs,
    ) -> "PlatonicCalculator":
        """Build a calculator from a ``mains/main_omol.py`` Lightning checkpoint.

        ``scale``, ``shift`` and ``element_refs`` are read from the checkpoint (they can
        still be overridden through ``kwargs``); all other keyword arguments go to
        :class:`PlatonicCalculator`. See :func:`load_force_field_checkpoint`.
        """
        model, meta = load_force_field_checkpoint(ckpt_path, config=config, use_ema=use_ema)
        kwargs.setdefault("scale", meta["scale"])
        kwargs.setdefault("shift", meta["shift"])
        kwargs.setdefault("element_refs", meta["element_refs"])
        return cls(model, **kwargs)

    # ------------------------------------------------------------------ helpers
    def _charge_spin(self, atoms) -> tuple[float, float]:
        return (float(atoms.info.get("charge", self.default_charge)),
                float(atoms.info.get("spin", self.default_spin)))

    def check_state(self, atoms, tol: float = 1e-15):
        """Extend ASE's change detection with ``atoms.info['charge'/'spin']``.

        ASE only compares positions, numbers, cell, pbc, initial charges and magmoms,
        so editing ``atoms.info`` alone would otherwise return cached results.
        """
        system_changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None and self._charge_spin(atoms) != self._charge_spin(self.atoms):
            system_changes.append("charge_spin")
        return system_changes

    def _atoms_to_data(self, atoms, requires_grad: bool) -> dict:
        if atoms.pbc.any():
            raise NotImplementedError(
                "PlatonicCalculator does not support periodic boundary conditions; the "
                "OMol force fields are trained on isolated molecules. Set atoms.pbc = False."
            )
        n = len(atoms)
        pos = torch.as_tensor(atoms.get_positions(), dtype=self.dtype, device=self.device)
        if requires_grad:
            pos.requires_grad_(True)
        charge, spin = self._charge_spin(atoms)
        return {
            "atomic_numbers": torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long,
                                              device=self.device),
            "pos": pos,
            "batch": torch.zeros(n, dtype=torch.long, device=self.device),
            "charge": torch.tensor([charge], dtype=torch.float32, device=self.device),
            "spin": torch.tensor([spin], dtype=torch.float32, device=self.device),
        }

    def _reference_energy(self, atomic_numbers: torch.Tensor) -> float:
        if self.element_refs is None:
            return 0.0
        return float(self.element_refs[atomic_numbers].sum())

    # ---------------------------------------------------------------- ASE hook
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        need_forces = "forces" in properties
        use_autograd = need_forces and self.force_mode == "autograd"
        data = self._atoms_to_data(self.atoms, requires_grad=use_autograd)

        with torch.set_grad_enabled(use_autograd):
            out = self.model(data)
            # Normalised model output -> eV. `energy` has shape (1,): one molecule.
            energy = out["energy"].reshape(-1)[0].to(torch.float64) * self.scale + self.shift
            if use_autograd:
                (grad,) = torch.autograd.grad(energy, data["pos"])
                forces = -grad
            elif need_forces:
                forces = out["forces"].to(torch.float64) * self.scale
            else:
                forces = None

        energy_ev = float(energy.detach()) + self._reference_energy(data["atomic_numbers"])
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        if forces is not None:
            self.results["forces"] = forces.detach().cpu().numpy().astype(np.float64)
