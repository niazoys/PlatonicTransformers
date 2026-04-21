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