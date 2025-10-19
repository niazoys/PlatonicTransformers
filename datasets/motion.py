import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

import pickle

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

tmp=os.path.dirname(current_file_path)

dataroot = os.path.join(tmp,'motion')

class Motion:
    """
    Motion Dataset

    """

    def __init__(self, partition, max_samples, delta_frame,all_joint_normals,data_dir):
        with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
            edges, X = pickle.load(f)
        V = []
        for i in range(len(X)):
            V.append(X[i][1:] - X[i][:-1])
            X[i] = X[i][:-1]


        N = X[0].shape[1]

        train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16]
        val_case_id = [3, 8, 11, 12, 15, 18]
        test_case_id = [6, 19, 21, 0, 22, 10]


        split_dir = os.path.join(data_dir, 'split.pkl')


        self.partition = partition

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pickle.load(f)
        except:
            np.random.seed(100)

            # sample 200 for each case (this was 100)
            itv = 100
            train_mapping = {}
            for i in train_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                train_mapping[i] = sampled
            val_mapping = {}
            for i in val_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                val_mapping[i] = sampled
            test_mapping = {}
            for i in test_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                test_mapping[i] = sampled

            with open(split_dir, 'wb') as f:
                pickle.dump((train_mapping, val_mapping, test_mapping), f)

            print('Generate and save split!')
            split = (train_mapping, val_mapping, test_mapping)

        if partition == 'train':
            mapping = split[0]
        elif partition == 'val':
            mapping = split[1]
        elif partition == 'test':
            mapping = split[2]
        else:
            raise NotImplementedError()

        each_len = max_samples // len(mapping) #what the hell is this?

        x_0, v_0, x_t, v_t = [], [], [], []
        for i in mapping:
            st = mapping[i][:each_len] # why are we doing this?
            cur_x_0 = X[i][st]
            cur_v_0 = V[i][st]
            cur_x_t = X[i][st + delta_frame]
            cur_v_t = V[i][st + delta_frame]
            x_0.append(cur_x_0)
            v_0.append(cur_v_0)
            x_t.append(cur_x_t)
            v_t.append(cur_v_t)
        x_0 = np.concatenate(x_0, axis=0)
        v_0 = np.concatenate(v_0, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        v_t = np.concatenate(v_t, axis=0)
        print('x_0', x_0.shape,'v_0' ,v_0.shape,'x_t', x_t.shape,'v_t', v_t.shape)

        print('Got {:d} samples!'.format(x_0.shape[0]))

        self.n_node = N

        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3] ??? what is this 3?
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        mole_idx = np.ones(N)
        self.mole_idx = torch.Tensor(mole_idx)

        if all_joint_normals:
            self.normal_vec = self.compute_normal_vectors(self.x_0,atom_edges)
        else:
            # Compute normal vectors only on the hip 
            # joint and use it for all joints
            joint_vec=self.x_0[:,0,:]-self.x_0[:,11,:]
            motion_vec=torch.tensor([0,0,1],dtype=joint_vec.dtype).repeat(self.x_0.shape[0],1)
            normal_vec=torch.cross(joint_vec,motion_vec)
            normal_vec=normal_vec/(torch.norm(normal_vec,dim=1,keepdim=True) + 1e-8)        
            self.normal_vec = normal_vec.unsqueeze(1).repeat(1,self.x_0.shape[1],1)


        self.cfg = self.sample_cfg()

    def compute_normal_vectors(self,x_0, atom_edges):
        """
        Compute normal vectors for each joint in a batch of samples, based on the adjacency matrix (atom_edges) and 
        joint positions (x_0). 
        
        Args:
        - x_0 (torch.Tensor): Input tensor of shape (batch_size, num_joints, 3) representing the 3D positions of joints.
        - atom_edges (torch.Tensor): Adjacency matrix of shape (num_joints, num_joints) indicating joint connections.
        
        Returns:
        - normal_vec (torch.Tensor): Tensor of shape (batch_size, num_joints, 3) representing the normal vectors for 
                                    each joint in each sample.
        """
        batch_size = x_0.shape[0]
        num_joints = x_0.shape[1]

        normal_vec = torch.zeros(batch_size, num_joints, 3, dtype=x_0.dtype)

        #TODO: Does it make sense to sometime add bit of random values on the other
        #TODO: two direction becuase clearly the motion is not completely in z direction
        motion_vec = torch.tensor([0, 0, 1], dtype=x_0.dtype).repeat(batch_size, 1)


        for i in range(num_joints):

            connected_joints = torch.nonzero(atom_edges[i]).squeeze().view(-1)

            if connected_joints.numel() == 0:
                continue

            # Compute the joint vector for each connected joint
            joint_vec = torch.zeros(batch_size, 3, dtype=x_0.dtype)
            for j in connected_joints:
                joint_vec += (x_0[:, i, :] - x_0[:, j, :])

            #TODO: Idk, I just felt like lets take avg but we can do other things also!           
            joint_vec /= len(connected_joints)
           
            normal_vec_batch = torch.cross(joint_vec, motion_vec)

            normal_vec_batch = normal_vec_batch / (torch.norm(normal_vec_batch, dim=1, keepdim=True) + 1e-8)

          
            normal_vec[:, i, :] = normal_vec_batch

        return normal_vec



    def sample_cfg(self):
        """
        Kinematics Decomposition
        What is going on ?
        """
        cfg = {}

        cfg['Stick'] = [(0, 11), (12, 13)]
        cfg['Stick'].extend([(2, 3), (7, 8), (17, 18), (24, 25)])

        cur_selected = []
        for _ in cfg['Stick']:
            cur_selected.append(_[0])
            cur_selected.append(_[1])

        cfg['Isolated'] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
        if len(cfg['Isolated']) == 0:
            cfg.pop('Isolated')

        return cfg

    def __getitem__(self, i):

        cfg = self.cfg

        edge_attr = self.edge_attr
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        edges = self.edges

        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                        stick_ind[m] = 1
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)  # [edge, 2]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        
        # Original retruns 
        #     return Data(
        #     loc=self.x_0[i],
        #     vel=self.v_0[i],
        #     y=self.x_t[i],
        #     edge_index=torch.tensor(self.edges)
        # )

        return Data(
            x=None,
            vel=self.v_0[i],
            edge_index=torch.tensor(self.edges),
            edge_attr=edge_attr,
            y=self.x_t[i],
            pos=self.x_0[i],
            normals=self.normal_vec[i]
            
        )

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


class MotionDataset:
    def __init__(self, batch_size=100, all_joint_normals=False, num_training_samples=2000):
        torch_geometric.seed.seed_everything(0)
        self.batch_size = batch_size
        self.all_joint_normals = all_joint_normals

        self.train_dataset = Motion(
            partition='train', 
            max_samples=num_training_samples, 
            delta_frame=30,
            all_joint_normals=self.all_joint_normals, 
            data_dir=dataroot
            )
        self.test_dataset = Motion(
            partition='test', 
            max_samples=600, 
            delta_frame=30, 
            all_joint_normals=self.all_joint_normals, 
            data_dir=dataroot
        )
        self.valid_dataset = Motion(
            partition='val', 
            max_samples=600, 
            delta_frame=30,
            all_joint_normals=self.all_joint_normals,  
            data_dir=dataroot
        )
   
    def train_loader(self):
        return PyGDataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            follow_batch=None
        )

    def val_loader(self):
        return PyGDataLoader(
            self.valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            follow_batch=None
        )

    def test_loader(self):
        return PyGDataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            follow_batch=None
        )




if __name__ == "__main__":
    dataset = MotionDataset(batch_size=1, all_joint_normals=True, num_training_samples=1100)
    train_loader = dataset.train_loader()
    
    print(len(train_loader))
    for data in train_loader:
        print(data)
        # data = to_po_point_cloud(data)
        break
    
    print('Done!')