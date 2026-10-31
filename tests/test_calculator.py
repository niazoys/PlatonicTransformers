"""CPU tests for `platonic_transformers.calculator.PlatonicCalculator`.

A tiny randomly initialised `PlatonicForceField` stands in for a trained model, so
the tests check the plumbing (ASE Atoms -> model input -> eV energies / eV/A forces)
rather than accuracy:

1. Shapes / dtypes of `energy` and `forces`, and the energy de-normalisation
   (`scale`, `shift`, per-element reference energies).
2. `forces="autograd"` is the exact negative gradient of the energy, checked against
   central finite differences.
3. Exact equivariance under the tetrahedral group (a 180 degree rotation about z)
   and translation invariance, for both force modes.
4. `from_checkpoint` round-trips a Lightning-style checkpoint, including EMA weights.
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402

from platonic_transformers.calculator import (  # noqa: E402
    PlatonicCalculator,
    load_force_field_checkpoint,
)
from platonic_transformers.models.platoformer.force_field import (  # noqa: E402
    PlatonicForceField,
    build_force_field,
)

TINY_MODEL_CFG = {
    "hidden_dim": 48,
    "num_heads": 12,          # must be a multiple of |G| = 12 for the tetrahedron
    "num_layers": 2,
    "solid_name": "tetrahedron",
    "dense_mode": False,
    "attention": True,
    "rope_sigma": 2.0,
    "rope_on_values": True,
    "qk_norm": True,
    "activation": "sin",
    "layer_scale_init_value": 1e-2,
    "chgspin_mode": "add",
    "chgspin_film": True,
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def model():
    return build_force_field(TINY_MODEL_CFG).eval()


@pytest.fixture
def methanol():
    atoms = molecule("CH3OH")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def test_energy_and_forces_shapes(model, methanol):
    calc = PlatonicCalculator(model, device="cpu")
    methanol.calc = calc
    e = methanol.get_potential_energy()
    f = methanol.get_forces()
    assert isinstance(e, float) and np.isfinite(e)
    assert f.shape == (len(methanol), 3)
    assert f.dtype == np.float64
    assert np.all(np.isfinite(f))
    assert calc.results["free_energy"] == e


def test_energy_denormalisation(model, methanol):
    refs = torch.zeros(100, dtype=torch.float64)
    refs[1], refs[6], refs[8] = -0.5, -1000.0, -2000.0  # H, C, O (eV)
    scale, shift = 1.433569, 0.25

    raw = PlatonicCalculator(model, device="cpu")
    full = PlatonicCalculator(model, device="cpu", scale=scale, shift=shift, element_refs=refs)
    methanol.calc = raw
    e_raw, f_raw = methanol.get_potential_energy(), methanol.get_forces()
    methanol.calc = full
    e_full, f_full = methanol.get_potential_energy(), methanol.get_forces()

    expected_ref = sum(float(refs[z]) for z in methanol.get_atomic_numbers())
    assert e_full == pytest.approx(e_raw * scale + shift + expected_ref, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(f_full, f_raw * scale, rtol=1e-6, atol=1e-8)


def test_autograd_forces_match_finite_differences(model, methanol):
    """-dE/dR from autograd vs central differences.

    `PlatonicForceField.forward` casts the fp64-reduced energy back to fp32, so the
    returned energy carries ~1e-7 relative rounding noise even when the calculator runs
    in fp64. A step of 5e-3 A keeps the finite-difference noise (~eps/h) well below the
    truncation error we tolerate.
    """
    calc = PlatonicCalculator(model, device="cpu", dtype=torch.float64, forces="autograd",
                              scale=1.433569)
    methanol.calc = calc
    forces = methanol.get_forces()

    # Translation invariance of the energy => autograd forces sum to zero exactly.
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)

    h = 5e-3
    for atom_idx in (0, 1):             # C and one H
        for k in range(3):
            fd = []
            for sign in (+1.0, -1.0):
                shifted = methanol.copy()
                pos = shifted.get_positions()
                pos[atom_idx, k] += sign * h
                shifted.set_positions(pos)
                shifted.calc = calc
                fd.append(shifted.get_potential_energy())
            numerical = -(fd[0] - fd[1]) / (2 * h)
            assert forces[atom_idx, k] == pytest.approx(numerical, rel=2e-2, abs=1e-4)


@pytest.mark.parametrize("force_mode", ["direct", "autograd"])
def test_equivariance_and_translation_invariance(model, methanol, force_mode):
    calc = PlatonicCalculator(model, device="cpu", dtype=torch.float64, forces=force_mode)
    methanol.calc = calc
    e0, f0 = methanol.get_potential_energy(), methanol.get_forces()

    # 180 degree rotation about z is an element of the tetrahedral rotation group,
    # so the (approximately equivariant) model is exactly equivariant to it.
    R = np.diag([-1.0, -1.0, 1.0])
    rotated = methanol.copy()
    rotated.set_positions(methanol.get_positions() @ R.T + np.array([1.5, -2.0, 0.7]))
    rotated.calc = calc
    e1, f1 = rotated.get_potential_energy(), rotated.get_forces()

    assert e1 == pytest.approx(e0, rel=1e-8, abs=1e-8)
    np.testing.assert_allclose(f1, f0 @ R.T, rtol=1e-7, atol=1e-8)


def test_charge_spin_from_atoms_info(model, methanol):
    calc = PlatonicCalculator(model, device="cpu")
    methanol.calc = calc
    e_neutral = methanol.get_potential_energy()

    cation = methanol.copy()
    cation.info["charge"], cation.info["spin"] = 1, 2
    cation.calc = calc
    assert cation.get_potential_energy() != pytest.approx(e_neutral, abs=1e-10)


def test_periodic_atoms_rejected(model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5], pbc=True)
    atoms.calc = PlatonicCalculator(model, device="cpu")
    with pytest.raises(NotImplementedError):
        atoms.get_potential_energy()


def test_invalid_force_mode(model):
    with pytest.raises(ValueError):
        PlatonicCalculator(model, device="cpu", forces="hessian")


def test_from_checkpoint_roundtrip(model, methanol, tmp_path):
    """Mimic the checkpoint layout written by mains/main_omol.py (OMolModel + EMACallback)."""
    ema_model = build_force_field(TINY_MODEL_CFG).eval()  # different random weights
    element_refs = torch.zeros(100, dtype=torch.float64)
    element_refs[1], element_refs[6], element_refs[8] = -13.6, -1029.8, -2042.6

    state_dict = {f"net.{k}": v for k, v in model.state_dict().items()}
    state_dict["scale"] = torch.tensor(1.433569)
    state_dict["shift"] = torch.tensor(0.0)
    state_dict["element_refs"] = element_refs
    ckpt = {
        "state_dict": state_dict,
        "hyper_parameters": {"config": {"model": dict(TINY_MODEL_CFG),
                                        "dataset": {"element_refs_path": None}}},
        "callbacks": {
            "EMACallback": {
                "decay": 0.99, "warmup_steps": 0,
                "ema_state": {f"net.{k}": v.clone().float()
                              for k, v in ema_model.named_parameters()},
            }
        },
    }
    path = tmp_path / "last.ckpt"
    torch.save(ckpt, path)

    # Raw weights reproduce `model`; EMA weights reproduce `ema_model`.
    for use_ema, reference in ((False, model), (True, ema_model)):
        calc = PlatonicCalculator.from_checkpoint(path, use_ema=use_ema, device="cpu")
        assert calc.scale == pytest.approx(1.433569)
        assert calc.element_refs is not None
        ref_calc = PlatonicCalculator(reference, device="cpu", scale=1.433569,
                                      element_refs=element_refs)
        a, b = methanol.copy(), methanol.copy()
        a.calc, b.calc = calc, ref_calc
        assert a.get_potential_energy() == pytest.approx(b.get_potential_energy(), rel=1e-6)
        np.testing.assert_allclose(a.get_forces(), b.get_forces(), rtol=1e-5, atol=1e-7)

    _, meta = load_force_field_checkpoint(path, use_ema=True)
    assert meta["ema_loaded"] is True
    assert meta["config"]["model"]["hidden_dim"] == TINY_MODEL_CFG["hidden_dim"]

    # A checkpoint without hyper_parameters needs an explicit config.
    bare = {"state_dict": state_dict}
    torch.save(bare, tmp_path / "bare.ckpt")
    with pytest.raises(ValueError):
        load_force_field_checkpoint(tmp_path / "bare.ckpt")
    m, _ = load_force_field_checkpoint(tmp_path / "bare.ckpt", config={"model": TINY_MODEL_CFG})
    assert isinstance(m, PlatonicForceField)
