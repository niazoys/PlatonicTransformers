import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch.utils.data import Dataset
from tqdm import tqdm


class QM9Dataset(Dataset):
    # Conversion factors for targets
    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414
    TOTAL_SIZE = 130831  # Total size of the dataset
    TRAIN_SIZE = 110000
    VAL_SIZE = 10000  # 110000:120000 for validation
    TEST_SIZE = 10831  # Remaining for test
    TARGETS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    UNCHARACTERIZED_URL = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root='./datasets/qm9_dataset', sdf_file='gdb9.sdf', csv_file='gdb9.sdf.csv', target=None, split=None, use_charges=False):
        self.root = root
        self.sdf_file = os.path.join(root, sdf_file)
        self.csv_file = os.path.join(root, csv_file)
        self.processed_file = os.path.join(root, 'processed_qm9_data.pkl')
        self.uncharacterized_file = os.path.join(root, 'uncharacterized.txt')
        self.qm9_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
        self.target_index = None if target is None else self.TARGETS.index(target)
        self.split = split  # Split can be 'train', 'val', or 'test'
        self.dataset_info = {'name': 'qm9',
                             'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
                             'atom_decoder': ['H', 'C', 'N', 'O', 'F']}
        self.use_charges = use_charges

        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        if os.path.isfile(self.processed_file):
            print("Loading processed data...")
            with open(self.processed_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.download_uncharacterized()
            self.ensure_data_downloaded()
            print("Processing data from scratch...")
            self.data = self.process()
            with open(self.processed_file, 'wb') as f:
                pickle.dump(self.data, f)

        if split:
            self.apply_split()
        
        self.dataset_to_pytorch()

    def apply_split(self):
        # Create the split based on the predefined sizes (seed used by DimeNet)
        random_state = np.random.RandomState(seed=42)
        perm = random_state.permutation(np.arange(self.TOTAL_SIZE))
        train_idx, val_idx, test_idx = perm[:self.TRAIN_SIZE], perm[self.TRAIN_SIZE:self.TRAIN_SIZE + self.VAL_SIZE], perm[self.TRAIN_SIZE + self.VAL_SIZE:]

        if self.split == 'train':
            self.data = [self.data[i] for i in train_idx]
        elif self.split == 'val':
            self.data = [self.data[i] for i in val_idx]
        elif self.split == 'test':
            self.data = [self.data[i] for i in test_idx]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
        
    def dataset_to_pytorch(self):
        for i in range(len(self.data)):
            self.data[i]['x'] = torch.from_numpy(self.data[i]['x'].astype(np.float32))
            self.data[i]['y'] = torch.from_numpy(self.data[i]['y'].astype(np.float32))
            self.data[i]['pos'] = torch.from_numpy(self.data[i]['pos'].astype(np.float32))
            self.data[i]['edge_attr'] = torch.from_numpy(self.data[i]['edge_attr'].astype(np.float32))
            self.data[i]['edge_index'] = torch.from_numpy(self.data[i]['edge_index'].astype(int))
            if self.use_charges:
                charges = torch.from_numpy(self.data[i]['charges'].astype(np.float32))
                self.data[i]['x'] = torch.cat([self.data[i]['x'], charges.unsqueeze(-1)], dim=-1)

    def download_uncharacterized(self):
        """Download the uncharacterized.txt file."""
        if not os.path.isfile(self.uncharacterized_file):
            print("Downloading uncharacterized.txt...")
            response = requests.get(self.UNCHARACTERIZED_URL)
            response.raise_for_status()  # Ensure the request was successful
            with open(self.uncharacterized_file, 'wb') as f:
                f.write(response.content)

    def read_uncharacterized_indices(self):
        """Read indices from uncharacterized.txt file."""
        # with open(self.uncharacterized_file, 'r') as file:
            # indices = [int(line.strip()) - 1 for line in file if line.strip().isdigit()]  # Adjusting indices to 0-based
        with open(self.uncharacterized_file, 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        return set(skip)

    def download_file(self, url, filename):
        local_filename = os.path.join(self.root, filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def extract_zip(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print(f"Extracted to {self.root}")

    def ensure_data_downloaded(self):
        if not os.path.isfile(self.sdf_file) or not os.path.isfile(self.csv_file):
            print(f"SDF or CSV file not found, downloading and extracting QM9 dataset...")
            zip_file_path = self.download_file(self.qm9_url, 'qm9.zip')
            self.extract_zip(zip_file_path)
        else:
            print("SDF and CSV files found, no need to download.")

    def process(self):
        # suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        suppl = Chem.SDMolSupplier(self.sdf_file, removeHs=False, sanitize=False)
        df = pd.read_csv(self.csv_file)
        raw_targets = df.iloc[:, 1:].values
        raw_targets = raw_targets.astype(np.float32)

        rearranged_targets = np.concatenate([raw_targets[:, 3:], raw_targets[:, :3]], axis=1)
        conversion_factors = np.array([
            1., 1., self.HAR2EV, self.HAR2EV, self.HAR2EV, 1., self.HAR2EV, self.HAR2EV, self.HAR2EV,
            self.HAR2EV, self.HAR2EV, 1., self.KCALMOL2EV, self.KCALMOL2EV, self.KCALMOL2EV,
            self.KCALMOL2EV, 1., 1., 1.
        ], dtype=np.float32)

        targets = rearranged_targets * conversion_factors

        atom_types = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        data_list = []

        skip_indices = self.read_uncharacterized_indices()


        for i, mol in enumerate(tqdm(suppl, desc="Processing Molecules")):
            if mol is None or i in skip_indices:  # Skip uncharacterized molecules
                continue
            # if mol is None: continue
            num_atoms = mol.GetNumAtoms()
            pos = np.array([mol.GetConformer().GetAtomPosition(j) for j in range(num_atoms)], dtype=np.float32)
            x = np.zeros((num_atoms, len(atom_types)), dtype=bool)  # one-hot encoding
            charges = np.zeros((num_atoms, ), dtype=int)  # integer charge encoding

            for j in range(num_atoms):
                atom = mol.GetAtomWithIdx(j)
                x[j, atom_types[atom.GetAtomicNum()]] = 1
                charges[j] = atom.GetFormalCharge()

            y = targets[i]
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol)

            # Initialize lists for edge indices and attributes
            edge_indices = []
            edge_attrs = []

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # Bond type one-hot encoding: single, double, triple, aromatic
                bond_type = [0, 0, 0, 0]
                if bond.GetBondType() == rdchem.BondType.SINGLE:
                    bond_type[0] = 1
                elif bond.GetBondType() == rdchem.BondType.DOUBLE:
                    bond_type[1] = 1
                elif bond.GetBondType() == rdchem.BondType.TRIPLE:
                    bond_type[2] = 1
                elif bond.GetBondType() == rdchem.BondType.AROMATIC:
                    bond_type[3] = 1

                edge_indices.append((start, end))
                edge_indices.append((end, start))  # Add reverse direction for undirected graph

                edge_attrs += [bond_type, bond_type]  # Same attributes for both directions

            # Convert edge data to tensors
            edge_index = np.array(edge_indices, dtype=int).T
            edge_attr = np.array(edge_attrs, dtype=bool)

            # Sorting edge_index by source node indices
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.target_index is not None:
            if len(item['y']) > 1:
                item['y'] = item['y'][self.target_index:self.target_index+1]
            else:
                # The item is already updated
                pass
        return item
       
    def NumAtomsSampler(self):
        class NumAtomsSampler(torch.nn.Module):
            def __init__(self, data):
                super(NumAtomsSampler, self).__init__()
                # Compute the histogram and probabilities during initialization
                num_atoms = torch.tensor([item['num_atoms'] for item in data], dtype=torch.float32)
                max_atoms = int(num_atoms.max().item())
                counts, _ = torch.histogram(num_atoms, bins=max_atoms, range=(1, max_atoms + 1))
                self.probabilities = counts.float() / counts.sum()
                
            def forward(self, num_samples):
                # Sample 'num_samples' integers according to the probability distribution
                sampled_bins = torch.multinomial(self.probabilities, num_samples, replacement=True) + 1  # +1 for 1-indexed bins
                return sampled_bins
        return NumAtomsSampler(self.data)

def collate_fn(batch):
    pos, x, y, batch_idx, edge_index_batch, edge_attr_batch = [], [], [], [], [], []
    cum_nodes = 0
    for i, item in enumerate(batch):
        num_nodes = item['x'].shape[0]
        pos.append(item['pos'])
        x.append(item['x'])
        y.append(item['y'])
        batch_idx.extend([i] * num_nodes)
        edge_index = item['edge_index'] + cum_nodes  # Offset node indices
        edge_index_batch.append(edge_index)
        edge_attr_batch.append(item['edge_attr'])
        cum_nodes += num_nodes
    pos = torch.cat(pos, dim=0)
    x = torch.cat(x, dim=0)
    y = torch.stack(y, dim=0)
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)
    edge_index = torch.cat(edge_index_batch, dim=1)
    edge_attr = torch.cat(edge_attr_batch, dim=0)

    return {'pos': pos, 'x': x, 'y': y, 'batch': batch_idx, 'edge_index': edge_index,'edge_attr': edge_attr}


# from torch.utils.data import DataLoader
# import time

# batch_size = 96
# dataset = QM9Dataset(target='alpha', split='train')
# # num_atoms_sampler = dataset.NumAtomsSampler()
# # sampled_bin = num_atoms_sampler(5)

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=0)

# t0 = time.time()
# posses = []
# for batch in dataloader:
#     x, y, pos, edge_index, edge_attr = batch['x'], batch['y'], batch['pos'], batch['edge_index'], batch['edge_attr']
#     posses.append(pos)
# posses = torch.cat(posses, dim=0)
# print(posses.std(dim=0).sqrt())
# print(posses.std().sqrt())
# t1 = time.time()
# print(t1-t0)