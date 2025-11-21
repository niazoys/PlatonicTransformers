import torch
from typing import Optional
from .utils import scatter_add


def lift_scalars(x, group):
    if x.ndim == 2:    # graph mode
        return x.unsqueeze(1).repeat(1, group.G, 1) # (N, C_hidden_g) -> (N, G, C_hidden_g)
    elif x.ndim == 3:  # dense mode
        return x.unsqueeze(2).repeat(1, 1, group.G, 1) # (B, N, C_hidden_g) -> (B, N, G, C_hidden_g)
    
def lift_vectors(x, group):
    frames = group.elements.type_as(x)  # (G, 3, 3)
    return torch.einsum('gji,...cj->...gci', frames, x).flatten(-2, -1)


def readout_scalars(x, group):
    return x.mean(dim=-2)  # (..., G * C) -> (..., G, C) -> (..., C)

def readout_vectors(x, group):
    x = x.unflatten(-1, (-1, group.dim))  # (..., G * C) -> (..., G, C, 3)
    frames = group.elements.type_as(x)  # (G, 3, 3)
    return torch.einsum('gij,...gcj->...ci', frames, x) / group.G  # frame transposed, result: (..., c, 3)

def lift(scalars, vectors, group):
    x_list = []
    if scalars is not None:
        x_list.append(lift_scalars(scalars, group))
    if vectors is not None:
        x_list.append(lift_vectors(vectors, group))
    return torch.cat(x_list, dim=-1).flatten(-2, -1)  # (..., G, C) -> (..., G * C)

def to_scalars_vectors(x, num_scalars, num_vectors, group):
    x = x.unflatten(-1, (group.G, -1))
    x_scalars, x_vectors = x.split([num_scalars, num_vectors * group.dim], dim=-1)  # (..., G * C) -> (..., G, C)
    scalars = readout_scalars(x_scalars, group)
    vectors = readout_vectors(x_vectors, group)
    return scalars, vectors  # (..., C), (..., C, 3)

def to_dense_and_mask(x: torch.Tensor, vec: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor]):
        """
        Converts sparse graph input tensors (x, pos, batch) to dense, padded tensors
        and generates an attention mask for dense mode. If inputs are already dense,
        it validates their shape and creates a full mask.

        Args:
            x (torch.Tensor): Node features. Can be (TotalN, C) for sparse or (B, N, C) for dense.
            pos (torch.Tensor): Node positions. Can be (TotalN, 3) for sparse or (B, N, 3) for dense.
            batch (Optional[torch.Tensor]): Batch index for each node, shape (TotalN,).
                                             Should be None if inputs are already dense.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dense_x (torch.Tensor): Padded node features of shape (B, N_max, C).
                - dense_pos (torch.Tensor): Padded node positions of shape (B, N_max, 3).
                - attention_mask (torch.Tensor): Boolean mask of shape (B, N_max),
                                                 True for real nodes, False for padded ones.
        Raises:
            ValueError: If input shapes are invalid for the given batch status.
        """

        if x is None and vec is None:
            raise ValueError("At least one of x or vec must be provided.")

        if (batch is None):  # input was already dense, only create mask (TODO: What if is just set to None?)
            if not (x.ndim == 3 and pos.ndim == 3 and x.shape[0] == pos.shape[0] and x.shape[1] == pos.shape[1]):
                 raise ValueError("If batch is None, x and pos must be [B, N, Dim] with matching B and N.")
            B, N, _ = x.shape
            attention_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            return x, vec, pos, attention_mask
        else:
            num_graphs = batch.max() + 1
            counts = torch.bincount(batch) # Number of nodes per graph
            max_nodes_per_graph = counts.max()
            
            dense_x = torch.zeros(num_graphs, max_nodes_per_graph, x.size(-1), device=x.device, dtype=x.dtype) if x is not None else None
            dense_vec = torch.zeros(num_graphs, max_nodes_per_graph, vec.size(-2), vec.size(-1), device=vec.device, dtype=vec.dtype) if vec is not None else None
            dense_pos = torch.zeros(num_graphs, max_nodes_per_graph, pos.size(-1), device=pos.device, dtype=pos.dtype)
            attention_mask = torch.zeros(num_graphs, max_nodes_per_graph, dtype=torch.bool, device=pos.device)
            
            for i in range(num_graphs):
                node_indices_in_sparse = (batch == i)
                num_nodes_in_this_graph = counts[i]
                
                if x is not None:
                    dense_x[i, :num_nodes_in_this_graph] = x[node_indices_in_sparse]
                if vec is not None:
                    dense_vec[i, :num_nodes_in_this_graph] = vec[node_indices_in_sparse]
                dense_pos[i, :num_nodes_in_this_graph] = pos[node_indices_in_sparse]
                attention_mask[i, :num_nodes_in_this_graph] = True # These are not padded
            
            return dense_x, dense_vec, dense_pos, attention_mask
        
def pool(x: torch.Tensor, 
         batch: torch.Tensor, 
         mask: Optional[torch.Tensor] = None, 
         avg_num_nodes: Optional[float] = None,
         dense_mode: bool = False,
         mean_aggregation: bool = True
         ) -> torch.Tensor:
    if dense_mode:
        x = x if mask is None else x * mask.unsqueeze(-1)  # [B, N_max, hidden_dim]
        x = x.sum(dim=1)
        # Normalize by number of nodes in each graph (or not, but then rescale by avg_num_nodes)
        if mean_aggregation and mask is not None:
            num_nodes = mask.sum(dim=1, keepdim=True).float()
        else:
            num_nodes = avg_num_nodes if avg_num_nodes is not None else 1.0
        x = x / num_nodes
    else:
        x = scatter_add(x, batch, dim_size=batch.max() + 1) / (avg_num_nodes if avg_num_nodes is not None else 1.0)
        # Normalize by number of nodes in each graph
        if mean_aggregation:
            num_nodes = scatter_add(torch.ones_like(batch, dtype=x.dtype), batch, dim_size=batch.max() + 1).unsqueeze(-1)
        else:
            num_nodes = avg_num_nodes if avg_num_nodes is not None else 1.0
        x = x / num_nodes
    return x