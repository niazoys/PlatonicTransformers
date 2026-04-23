"""
QM9 molecular-generation evaluation utilities.

We support two evaluation protocols in parallel:

1. EDM protocol (Hoogeboom et al. 2022, arXiv:2203.17003).
   Bonds are inferred using the distance-lookup table in qm9_bond_analyze
   (the same table used by check_stability), atoms are kept with hydrogens,
   and SMILES are canonical but non-isomeric. See `BasicMolecularMetrics`
   and `build_molecule`.

2. Zatom-1 protocol (arXiv:2602.22251).
   Bonds are re-perceived by RDKit's PDB reader from a PDB representation
   of the atoms+coords, hydrogens are stripped (`removeHs=True`), and
   SMILES are isomeric. Additionally, PoseBusters `config="mol"` is run on
   the RDKit mols that survive sanitization. See `ZatomMolecularMetrics`
   and `build_molecule_zatom_style`.

   DEVIATION FROM THE PUBLISHED ZATOM-1 PIPELINE:
   The Zatom-1 reference code (github.com/Zatom-AI/zatom) writes the PDB
   file via `pymatgen.core.Molecule.to(fmt='pdb')`, which internally calls
   `BabelMolAdaptor` and therefore REQUIRES the OpenBabel Python bindings.
   OpenBabel is not pip-installable without building from source and is
   not available on our Snellius / IVI clusters. To stay semantically
   faithful we:
     (a) build an RDKit RWMol with the atoms and the 3D conformer but NO
         bonds (mirroring the bond-free atom list that pymatgen writes),
     (b) dump that to a PDB string via `Chem.MolToPDBBlock`, which produces
         valid ATOM/HETATM records and no CONECT records,
     (c) read it back with `Chem.MolFromPDBBlock(..., removeHs=True,
         proximityBonding=True)`.
   The bond perception then runs inside RDKit's PDB reader from ATOM
   records (proximity bonding) — identical to what happens in the Zatom-1
   pipeline once the PDB text reaches RDKit. The only difference is the
   PDB writer used to produce the intermediate string. We believe this
   does not meaningfully change the metric because neither path emits
   CONECT records and both rely on RDKit's proximity perception, but
   numbers may differ from published Zatom-1 values at the sub-percent
   level. If an exact match is ever needed, install OpenBabel and swap
   in `pymatgen.core.Molecule.to(fmt='pdb')`.
"""

import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
from rdkit import Chem

from platonic_transformers.datasets.qm9_bond_analyze import geom_predictor, get_bond_order

#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_info = dataset_info
        # dataset_smiles_list is expected to be the set of canonical SMILES obtained
        # by running the training molecules through the same build_molecule + mol2smiles
        # pipeline that generated molecules go through. This is how EDM (Hoogeboom et
        # al. 2022) reports novelty; computing training SMILES via the SDF path
        # instead produces a ~15% false-novelty floor because the two bond-perception
        # routes canonicalize differently.
        self.dataset_smiles_set = set(dataset_smiles_list) if dataset_smiles_list else None

    def compute_validity(self, generated):
        """Compute validity following the EDM protocol (Hoogeboom et al. 2022).

        Bond graph is built from 3D coordinates + atom types using the bond-
        length lookup table in qm9_bond_analyze (same as check_stability).
        Molecule is then sanitized; largest connected fragment's canonical
        SMILES is retained as the valid representative.
        """
        valid = []
        for graph in generated:
            mol = build_molecule(*graph[:2], self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is None:
                continue
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            if smiles is not None:
                valid.append(smiles)
        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        """Fraction of unique generated SMILES not in the training set."""
        if not self.dataset_smiles_set:
            return list(unique), 1.0
        novel = [s for s in unique if s not in self.dataset_smiles_set]
        return novel, len(novel) / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_set is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique


def compute_training_smiles(dataset, dataset_info, cache_path=None, show_progress=True):
    """Run the training dataset through the EDM build_molecule pipeline to produce
    a canonical SMILES set. Results are cached on disk to avoid the ~few-minute
    recomputation cost on every run.

    Args:
        dataset: iterable of dict items with keys "pos", "x" (features: one-hot atom
                 type in first 5 columns, optional charge as 6th).
        dataset_info: qm9 dataset info dict with atom_decoder.
        cache_path: optional path for pickle cache. If the file exists and was built
                    for the same dataset size, it is loaded directly.
        show_progress: show a tqdm bar while building.

    Returns:
        set of canonical SMILES strings representing the training molecules as
        perceived by the evaluator's bond-inference pipeline.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("size") == len(dataset):
            return cached["smiles_set"]

    smiles_set = set()
    iterator = tqdm(dataset, desc="building training SMILES") if show_progress else dataset
    for item in iterator:
        at = item["x"][:, :5].argmax(dim=-1)
        mol = build_molecule(item["pos"], at, dataset_info)
        s = mol2smiles(mol)
        if s is None:
            continue
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        s = mol2smiles(largest)
        if s is not None:
            smiles_set.add(s)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"size": len(dataset), "smiles_set": smiles_set}, f)
    return smiles_set


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol


def build_molecule_zatom_style(positions, atom_types, dataset_info, remove_hs: bool = False):
    """Zatom-1 bond inference: PDB round-trip.

    Matches Zatom-1 protocol. `remove_hs` defaults to False to match
    `configs/model/zatom.yaml` default in the Zatom-1 reference code.
    (Hydrogens are retained through bond perception and PoseBusters.)

    See the module docstring for the deviation from the published pipeline
    (pymatgen PDB writer not available -> using RDKit MolToPDBBlock). Bond
    perception is still done by RDKit's proximityBonding from the PDB
    ATOM records, matching Zatom-1 semantically.
    """
    atom_decoder = dataset_info["atom_decoder"]
    # Build a bond-less RDKit mol with a 3D conformer so MolToPDBBlock has
    # something to dump. No bonds are added here — PDB proximity bonding
    # will perceive them on readback.
    try:
        bare = Chem.RWMol()
        for at in atom_types:
            bare.AddAtom(Chem.Atom(atom_decoder[at.item()]))
        conf = Chem.Conformer(int(positions.shape[0]))
        coords = positions.detach().cpu().numpy()
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        bare.AddConformer(conf, assignId=True)

        pdb_text = Chem.MolToPDBBlock(bare.GetMol())
        mol = Chem.MolFromPDBBlock(
            pdb_text,
            removeHs=remove_hs,
            proximityBonding=True,
        )
        return mol
    except Exception:
        return None


def _largest_frag_smiles(mol, isomeric):
    """Sanitize, take the largest connected fragment, return canonical SMILES.

    isomeric=False reproduces EDM's Chem.MolToSmiles default (canonical but
    not isomeric). isomeric=True reproduces Zatom-1's isomericSmiles=True.
    Returns None if the mol cannot be sanitized.
    """
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest)
        return Chem.MolToSmiles(largest, isomericSmiles=isomeric)
    except (ValueError, RuntimeError):
        return None


class ZatomMolecularMetrics(object):
    """Validity / uniqueness / novelty following the Zatom-1 protocol.

    Identical API to BasicMolecularMetrics but uses build_molecule_zatom_style
    for bond perception and isomericSmiles=True. Novelty is computed against
    a SMILES set derived by passing the training set through the SAME pipeline
    (see compute_training_smiles_zatom), so values are internally consistent
    with the EDM implementation's novelty being self-consistent.
    """

    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info
        self.dataset_smiles_set = set(dataset_smiles_list) if dataset_smiles_list else None

    def compute_validity(self, generated):
        """Returns (valid_smiles_list, valid_mol_list, validity).

        valid_mol_list contains the LARGEST-FRAGMENT mol with its original 3D
        conformer preserved, not a re-parse from SMILES — PoseBusters needs
        3D coordinates for every one of its geometric checks (bond lengths,
        bond angles, steric clash, ring flatness, internal energy).
        """
        valid_smiles, valid_mols = [], []
        for graph in generated:
            mol = build_molecule_zatom_style(*graph[:2], self.dataset_info)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except (ValueError, RuntimeError):
                continue
            try:
                frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest)
                s = Chem.MolToSmiles(largest, isomericSmiles=True)
            except (ValueError, RuntimeError):
                continue
            if s is None:
                continue
            valid_smiles.append(s)
            valid_mols.append(largest)  # keep 3D conformer for PoseBusters
        return valid_smiles, valid_mols, len(valid_smiles) / len(generated)

    def compute_uniqueness(self, valid_smiles):
        if not valid_smiles:
            return [], 0.0
        unique = list(set(valid_smiles))
        return unique, len(unique) / len(valid_smiles)

    def compute_novelty(self, unique, n_valid: int):
        """Zatom-1 definition: `|novel| / |valid|` (not `|novel| / |unique|`).

        See zatom/eval/molecule_generation.py: `novel_rate = len(novel_smiles)
        / len(valid_smiles)`. This differs from the EDM convention used in
        BasicMolecularMetrics, which divides by `len(unique)`. Keep both
        conventions intact so our `*_zatom` numbers are directly comparable
        to Zatom-1's Table and our `*_edm` numbers are directly comparable
        to Hoogeboom et al. 2022.
        """
        if not self.dataset_smiles_set or not unique:
            return list(unique), 1.0
        novel = [s for s in unique if s not in self.dataset_smiles_set]
        return novel, len(novel) / max(1, n_valid)

    def evaluate(self, generated):
        valid_smiles, valid_mols, validity = self.compute_validity(generated)
        if valid_smiles:
            unique, uniqueness = self.compute_uniqueness(valid_smiles)
            if self.dataset_smiles_set is not None:
                _, novelty = self.compute_novelty(unique, n_valid=len(valid_smiles))
            else:
                novelty = 0.0
        else:
            uniqueness = 0.0
            novelty = 0.0
        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "valid_mols": valid_mols,
        }


def compute_training_smiles_zatom(dataset, dataset_info, cache_path=None, show_progress=True):
    """Zatom-1 companion to compute_training_smiles.

    Runs the training set through build_molecule_zatom_style + isomeric
    largest-fragment SMILES so the reference set for the Zatom-style
    novelty metric is canonicalized by the same rules as the generated
    SMILES.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("size") == len(dataset):
            return cached["smiles_set"]

    smiles_set = set()
    iterator = tqdm(dataset, desc="building training SMILES (zatom)") if show_progress else dataset
    for item in iterator:
        at = item["x"][:, :5].argmax(dim=-1)
        mol = build_molecule_zatom_style(item["pos"], at, dataset_info)
        s = _largest_frag_smiles(mol, isomeric=True)
        if s is not None:
            smiles_set.add(s)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"size": len(dataset), "smiles_set": smiles_set}, f)
    return smiles_set


def run_posebusters(rdkit_mols, max_molecules: int = 1000, seed: int = 0):
    """Run PoseBusters (config='mol') on a list of sanitized RDKit mols.

    PoseBusters' `internal_energy` check dominates runtime: per molecule it
    calls RDKit ETKDG to generate an ensemble of conformers plus UFF
    optimization, which can take several seconds per molecule — running on
    the full 10k-molecule validation pool stalls training for an hour+ per
    validation. We therefore sub-sample to `max_molecules` molecules by
    default (Zatom-1 runs on the full pool; this is a speed trade-off we
    accept for per-validation cost, documented in the module docstring).
    Validity / uniqueness / novelty / stability are still computed over the
    full pool.

    Pass `max_molecules=-1` (or any non-positive value) to disable the
    subsample and run on every mol. Intended for the final test evaluation
    where apples-to-apples paper-table numbers matter more than speed.

    The headline `posebusters_pass_rate` is `df.all(axis=1).mean()` over
    every column returned by PoseBusters, matching Zatom-1's
    `pb_metrics_dict["posebusters_rate"] = pb_metrics.all(axis=1)` on
    `zatom/eval/molecule_generation.py`. That includes the three plumbing
    columns (`mol_pred_loaded`, `sanitization`, `inchi_convertible`) and
    every substantive check (`all_atoms_connected`, `bond_lengths`,
    `bond_angles`, `internal_steric_clash`, `aromatic_ring_flatness`,
    `non-aromatic_ring_non-flatness` if available, `double_bond_flatness`,
    `internal_energy`).

    Per-column pass rates are logged as `posebusters/<column>_rate` for
    every column PoseBusters returns, so downstream analysis can see the
    plumbing and per-check breakdowns separately.
    If the list is empty, returns zeros so logging never blows up.
    """
    metrics = {"posebusters_pass_rate": 0.0, "posebusters/num_sampled": 0}
    if not rdkit_mols:
        return metrics

    # Sub-sample deterministically (seeded) when the pool is too large.
    # max_molecules<=0 means "no sub-sample" (use every mol).
    if max_molecules > 0 and len(rdkit_mols) > max_molecules:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rdkit_mols), size=max_molecules, replace=False)
        rdkit_mols = [rdkit_mols[i] for i in idx]
    metrics["posebusters/num_sampled"] = len(rdkit_mols)

    from posebusters import PoseBusters  # imported lazily to avoid startup cost

    pb = PoseBusters(config="mol")
    df = pb.bust(mol_pred=rdkit_mols)
    # Each PoseBusters check returns either a bool or a numeric "score" where
    # the boolean pass/fail flag is stored alongside. For checks whose kernel
    # errored on some molecules (most commonly `internal_energy` — its ETKDG
    # conformer generator sometimes can't re-embed our structures) pandas
    # silently promotes the column to object dtype containing {True, False,
    # NaN}. Picking only `dtype == bool` therefore drops exactly those rows
    # from the report. Keep any column whose non-NaN values are a subset of
    # {True, False}.
    def _is_pass_fail_column(series) -> bool:
        if series.dtype == bool or series.dtype.name == "boolean":
            return True
        vals = series.dropna().unique()
        return len(vals) > 0 and set(vals).issubset({True, False})

    check_cols = [c for c in df.columns if _is_pass_fail_column(df[c])]
    for c in check_cols:
        # .mean() on a boolean series ignores NaN by default via skipna; cast
        # the boolean-with-NaN "object" columns to float to get the same.
        col = df[c]
        if col.dtype != bool and col.dtype.name != "boolean":
            col = col.astype("boolean").astype("float")
        metrics[f"posebusters/{c}_rate"] = float(col.mean())
    if check_cols:
        # Aggregate pass rate: every check must pass. NaNs are treated as fail
        # (we don't know if they'd pass or not — conservative choice) to avoid
        # inflating the headline.
        passes = df[check_cols].fillna(False)
        metrics["posebusters_pass_rate"] = float(passes.all(axis=1).mean())
    return metrics


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)