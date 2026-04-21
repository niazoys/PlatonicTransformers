import os
import pickle
import tempfile

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from platonic_transformers.datasets.qm9_bond_analyze import geom_predictor, get_bond_order

#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        # if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            # self.dataset_smiles_list = retrieve_qm9_smiles(
                # self.dataset_info)

    def compute_validity(self, generated):
        """
        Compute validity of a list of generated molecules.
        Uses RDKit's rdDetermineBonds to infer bond orders from 3D coordinates.
        """
        valid = []

        for graph in generated:
            mol = build_molecule_from_xyz(*graph[:2], self.dataset_info)
            if mol is None:
                continue

            smiles = mol2smiles(mol)
            if smiles is not None:
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
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(mol)


def build_molecule_from_xyz(positions, atom_types, dataset_info):
    """
    Build an RDKit molecule from 3D coordinates + atom types.

    Uses RDKit's rdDetermineBonds.DetermineBonds which infers both connectivity
    and bond orders from interatomic distances and valence constraints. No
    external dependency on OpenBabel/pymatgen.
    """
    atom_decoder = dataset_info["atom_decoder"]
    atomic_symbols = [atom_decoder[atom.item()] for atom in atom_types]
    pos_numpy = positions.detach().cpu().numpy()

    # Build a minimal XYZ block (first line = atom count, second = comment, then rows).
    lines = [str(len(atomic_symbols)), ""]
    for sym, (x, y, z) in zip(atomic_symbols, pos_numpy):
        lines.append(f"{sym} {float(x):.6f} {float(y):.6f} {float(z):.6f}")
    xyz_block = "\n".join(lines) + "\n"

    try:
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        # DetermineBonds mutates mol in place: adds bonds + bond orders.
        # QM9 molecules are all neutral.
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return mol
    except Exception:
        return None


# The original build_molecule and build_xae_molecule functions are kept below
# in case they are used by other parts of your project, but they are no longer
# called by the BasicMolecularMetrics.evaluate() flow.

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