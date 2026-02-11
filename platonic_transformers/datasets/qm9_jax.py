"""
JAX/NumPy-compatible QM9 dataset without PyTorch dependencies.

This module provides a QM9 dataset implementation that works purely with NumPy arrays,
making it suitable for JAX-based training pipelines.
"""

import os
import pickle
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm


@dataclass
class Batch:
    """Batch of graph data, mimicking PyG batch interface for compatibility."""
    x: np.ndarray          # Node features [total_nodes, feat_dim]
    pos: np.ndarray        # Positions [total_nodes, 3]
    batch: np.ndarray      # Batch indices [total_nodes]
    y: np.ndarray          # Targets [batch_size] or [batch_size, num_targets]
    num_nodes: int         # Total number of nodes in batch
    num_graphs: int        # Number of graphs in batch
    
    def numpy(self):
        """Return self - already numpy arrays."""
        return self


class QM9DatasetJax:
    """
    QM9 Dataset with pure NumPy arrays for JAX compatibility.
    
    This dataset loads and processes the QM9 molecular dataset without
    requiring PyTorch. It provides direct numpy array access suitable
    for JAX training pipelines.
    """
    
    # Conversion factors for targets
    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414
    TOTAL_SIZE = 130831
    TRAIN_SIZE = 110000
    VAL_SIZE = 10000
    TEST_SIZE = 10831
    
    TARGETS = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 
        'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 
        'H_atom', 'G_atom', 'A', 'B', 'C'
    ]
    
    UNCHARACTERIZED_URL = 'https://ndownloader.figshare.com/files/3195404'
    QM9_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'

    def __init__(
        self, 
        root: str = './datasets/qm9_dataset', 
        target: Optional[str] = None,
        split: Optional[str] = None,
        use_charges: bool = False
    ):
        """
        Initialize QM9 dataset.
        
        Args:
            root: Root directory for dataset storage
            target: Target property name (e.g., 'alpha', 'homo', 'U0')
            split: Data split ('train', 'val', 'test', or None for full dataset)
            use_charges: Whether to include formal charges in node features
        """
        self.root = os.path.expanduser(root)
        self.sdf_file = os.path.join(self.root, 'gdb9.sdf')
        self.csv_file = os.path.join(self.root, 'gdb9.sdf.csv')
        self.processed_file = os.path.join(self.root, 'processed_qm9_data_numpy.pkl')
        self.uncharacterized_file = os.path.join(self.root, 'uncharacterized.txt')
        
        self.target_index = None if target is None else self.TARGETS.index(target)
        self.split = split
        self.use_charges = use_charges
        
        self.dataset_info = {
            'name': 'qm9',
            'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
            'atom_decoder': ['H', 'C', 'N', 'O', 'F']
        }

        os.makedirs(self.root, exist_ok=True)

        if os.path.isfile(self.processed_file):
            print("Loading processed data...")
            with open(self.processed_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self._download_uncharacterized()
            self._ensure_data_downloaded()
            print("Processing data from scratch...")
            self.data = self._process()
            with open(self.processed_file, 'wb') as f:
                pickle.dump(self.data, f)

        if split:
            self._apply_split()
        
        # Ensure arrays are float32 for JAX
        self._convert_to_float32()

    def _apply_split(self):
        """Apply train/val/test split."""
        random_state = np.random.RandomState(seed=42)
        perm = random_state.permutation(np.arange(self.TOTAL_SIZE))
        
        train_idx = perm[:self.TRAIN_SIZE]
        val_idx = perm[self.TRAIN_SIZE:self.TRAIN_SIZE + self.VAL_SIZE]
        test_idx = perm[self.TRAIN_SIZE + self.VAL_SIZE:]

        if self.split == 'train':
            self.data = [self.data[i] for i in train_idx]
        elif self.split == 'val':
            self.data = [self.data[i] for i in val_idx]
        elif self.split == 'test':
            self.data = [self.data[i] for i in test_idx]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
    
    def _convert_to_float32(self):
        """Convert all arrays to float32 for JAX compatibility."""
        for i in range(len(self.data)):
            self.data[i]['x'] = self.data[i]['x'].astype(np.float32)
            self.data[i]['y'] = self.data[i]['y'].astype(np.float32)
            self.data[i]['pos'] = self.data[i]['pos'].astype(np.float32)
            self.data[i]['edge_attr'] = self.data[i]['edge_attr'].astype(np.float32)
            self.data[i]['edge_index'] = self.data[i]['edge_index'].astype(np.int32)
            if self.use_charges:
                charges = self.data[i]['charges'].astype(np.float32)
                self.data[i]['x'] = np.concatenate(
                    [self.data[i]['x'], charges[:, np.newaxis]], axis=-1
                )

    def _download_uncharacterized(self):
        """Download the uncharacterized.txt file."""
        if not os.path.isfile(self.uncharacterized_file):
            print("Downloading uncharacterized.txt...")
            response = requests.get(self.UNCHARACTERIZED_URL)
            response.raise_for_status()
            with open(self.uncharacterized_file, 'wb') as f:
                f.write(response.content)

    def _read_uncharacterized_indices(self) -> set:
        """Read indices from uncharacterized.txt file."""
        with open(self.uncharacterized_file, 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        return set(skip)

    def _download_file(self, url: str, filename: str) -> str:
        """Download a file from URL."""
        local_filename = os.path.join(self.root, filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def _extract_zip(self, file_path: str):
        """Extract a zip file."""
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print(f"Extracted to {self.root}")

    def _ensure_data_downloaded(self):
        """Ensure raw data files exist."""
        if not os.path.isfile(self.sdf_file) or not os.path.isfile(self.csv_file):
            print("SDF or CSV file not found, downloading and extracting QM9 dataset...")
            zip_file_path = self._download_file(self.QM9_URL, 'qm9.zip')
            self._extract_zip(zip_file_path)
        else:
            print("SDF and CSV files found, no need to download.")

    def _process(self) -> List[Dict]:
        """Process raw QM9 data into list of dictionaries."""
        suppl = Chem.SDMolSupplier(self.sdf_file, removeHs=False, sanitize=False)
        df = pd.read_csv(self.csv_file)
        raw_targets = df.iloc[:, 1:].values.astype(np.float32)

        # Rearrange targets and apply conversion factors
        rearranged_targets = np.concatenate(
            [raw_targets[:, 3:], raw_targets[:, :3]], axis=1
        )
        conversion_factors = np.array([
            1., 1., self.HAR2EV, self.HAR2EV, self.HAR2EV, 1., self.HAR2EV, 
            self.HAR2EV, self.HAR2EV, self.HAR2EV, self.HAR2EV, 1., 
            self.KCALMOL2EV, self.KCALMOL2EV, self.KCALMOL2EV,
            self.KCALMOL2EV, 1., 1., 1.
        ], dtype=np.float32)

        targets = rearranged_targets * conversion_factors

        atom_types = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        data_list = []
        skip_indices = self._read_uncharacterized_indices()

        for i, mol in enumerate(tqdm(suppl, desc="Processing Molecules")):
            if mol is None or i in skip_indices:
                continue
                
            num_atoms = mol.GetNumAtoms()
            pos = np.array(
                [mol.GetConformer().GetAtomPosition(j) for j in range(num_atoms)],
                dtype=np.float32
            )
            
            # One-hot encoding for atom types
            x = np.zeros((num_atoms, len(atom_types)), dtype=np.float32)
            charges = np.zeros(num_atoms, dtype=np.int32)

            for j in range(num_atoms):
                atom = mol.GetAtomWithIdx(j)
                x[j, atom_types[atom.GetAtomicNum()]] = 1.0
                charges[j] = atom.GetFormalCharge()

            y = targets[i]
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol)

            # Process edges
            edge_indices = []
            edge_attrs = []

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # Bond type one-hot encoding: single, double, triple, aromatic
                bond_type = [0.0, 0.0, 0.0, 0.0]
                if bond.GetBondType() == rdchem.BondType.SINGLE:
                    bond_type[0] = 1.0
                elif bond.GetBondType() == rdchem.BondType.DOUBLE:
                    bond_type[1] = 1.0
                elif bond.GetBondType() == rdchem.BondType.TRIPLE:
                    bond_type[2] = 1.0
                elif bond.GetBondType() == rdchem.BondType.AROMATIC:
                    bond_type[3] = 1.0

                edge_indices.append((start, end))
                edge_indices.append((end, start))
                edge_attrs += [bond_type, bond_type]

            # Convert edge data to arrays
            edge_index = np.array(edge_indices, dtype=np.int32).T if edge_indices else np.zeros((2, 0), dtype=np.int32)
            edge_attr = np.array(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, 4), dtype=np.float32)

            # Sort edge_index by source node indices
            if edge_index.shape[1] > 0:
                sort_indices = np.lexsort((edge_index[0, :], edge_index[1, :]))
                edge_index = edge_index[:, sort_indices]
                edge_attr = edge_attr[sort_indices]

            data_list.append({
                'pos': pos,
                'x': x,
                'y': y,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'name': name,
                'smiles': smiles,
                'idx': i,
                'num_atoms': num_atoms,
                'charges': charges
            })

        return data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx].copy()
        if self.target_index is not None:
            if len(item['y']) > 1:
                item['y'] = item['y'][self.target_index:self.target_index + 1]
        return item


def collate_fn_jax(batch: List[Dict]) -> Batch:
    """
    Collate function for batching graph data into JAX-compatible format.
    
    Args:
        batch: List of graph dictionaries
        
    Returns:
        Batch object with concatenated arrays
    """
    pos_list, x_list, y_list, batch_idx_list = [], [], [], []
    cum_nodes = 0
    
    for i, item in enumerate(batch):
        num_nodes = item['x'].shape[0]
        pos_list.append(item['pos'])
        x_list.append(item['x'])
        y_list.append(item['y'])
        batch_idx_list.extend([i] * num_nodes)
        cum_nodes += num_nodes
    
    return Batch(
        pos=np.concatenate(pos_list, axis=0),
        x=np.concatenate(x_list, axis=0),
        y=np.stack(y_list, axis=0),
        batch=np.array(batch_idx_list, dtype=np.int32),
        num_nodes=cum_nodes,
        num_graphs=len(batch)
    )


class DataLoaderJax:
    """
    Simple data loader for JAX training that mimics PyTorch DataLoader interface.
    
    This provides an iterator over batched graph data without PyTorch dependencies.
    """
    
    def __init__(
        self,
        dataset: QM9DatasetJax,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,  # Kept for interface compatibility (unused)
        seed: int = 0
    ):
        """
        Initialize data loader.
        
        Args:
            dataset: QM9DatasetJax instance
            batch_size: Number of graphs per batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop the last incomplete batch
            num_workers: Unused, kept for interface compatibility
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        
    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(indices):
                if self.drop_last:
                    break
                end_idx = len(indices)
            
            batch_indices = indices[start_idx:end_idx]
            batch = [self.dataset[i] for i in batch_indices]
            
            yield collate_fn_jax(batch)
