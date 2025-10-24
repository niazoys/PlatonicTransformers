import torch

def scatter_add(src, index, dim_size):
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    return out.scatter_add_(0, index_expanded, src)