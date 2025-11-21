import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# Import the pre-computed group data
from .groups import PLATONIC_GROUPS

class PlatonicLinear(nn.Module):
    """
    A Linear layer constrained to be a group convolution over a Platonic Solid group.
    This version includes a corrected initialization scheme to preserve variance.
    """
    def __init__(self, in_features: int, out_features: int, solid: str, bias: bool = True):
        super().__init__()
        
        if solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(f"Solid '{solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}")
        
        group = PLATONIC_GROUPS[solid.lower()]
        self.G = group.G
        self.in_features = in_features
        self.out_features = out_features

        if in_features % self.G != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by the group order {self.G} for solid '{solid}'")
        if out_features % self.G != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by the group order {self.G} for solid '{solid}'")
            
        self.in_channels = in_features // self.G
        self.out_channels = out_features // self.G

        self.kernel = nn.Parameter(torch.empty(self.G, self.out_channels, self.in_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('cayley_table', group.cayley_table)
        self.register_buffer('inverse_indices', group.inverse_indices)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the kernel and bias with variance-preserving scaling.
        
        Standard initializers (like Kaiming) fail to correctly infer the
        effective fan-in of the full weight matrix. We must calculate it
        manually as (group_size * in_channels_per_group).
        """
        # Calculate the effective fan-in for the full weight matrix.
        fan_in = self.G * self.in_channels
        
        # Initialize the kernel from a normal distribution. The std is calculated
        # to ensure the output variance is approximately equal to the input variance.
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.kernel, mean=0.0, std=std)

        if self.bias is not None:
            # Initialize bias using the same correct fan-in.
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def get_weight(self) -> Tensor:
        """
        Constructs the full [G*O, G*I] weight matrix from the fundamental kernel.
        """
        device = self.kernel.device
        h_indices = torch.arange(self.G, device=device).view(self.G, 1)
        g_indices = torch.arange(self.G, device=device).view(1, self.G)

        inv_g_indices = self.inverse_indices[g_indices]
        kernel_group_idx = self.cayley_table[inv_g_indices, h_indices]
        
        expanded_kernel = self.kernel[kernel_group_idx]
        weight = expanded_kernel.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return weight

    def forward(self, x: Tensor) -> Tensor:
        """Applies the group-equivariant linear transformation."""
        weight = self.get_weight()
        output = F.linear(x, weight, None)
        
        if self.bias is not None:
            output_shape = output.shape
            output = output.view(*output_shape[:-1], self.G, self.out_channels)
            output = output + self.bias
            output = output.view(output_shape)
            
        return output
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(G={self.G}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
    




# Equivariance test

def run_equivariance_test(solid_name: str):
    group = PLATONIC_GROUPS[solid_name]
    G = group.G
    
    I, O, B = 4, 8, 2
    in_feats = G * I
    out_feats = G * O

    print(f"Initializing PlatonicLinear(solid='{solid_name}')")
    print(f"  (G={G}, I={I}, O={O}) -> in_features={in_feats}, out_features={out_feats}")
    layer = PlatonicLinear(in_features=in_feats, out_features=out_feats, solid=solid_name)
    
    torch.manual_seed(42)
    input_signal = torch.randn(B, in_feats)
    
    print("--- Testing Right-Equivariance Property ---")
    
    all_tests_passed = True
    original_output = layer(input_signal)

    for h in range(G):
        # transform_indices = group.cayley_table[:, h]
        transform_indices = group.cayley_table[h, :]
        
        if len(torch.unique(transform_indices)) != G:
            print(f"[!] Cayley table error at h={h}. Column is not a permutation!")
            all_tests_passed = False
            break

        input_unflattened = input_signal.view(B, G, I)
        transformed_unflattened = input_unflattened[:, transform_indices]
        transformed_input = transformed_unflattened.reshape(B, in_feats)
        output_lhs = layer(transformed_input)

        original_output_unflattened = original_output.view(B, G, O)
        transformed_output_unflattened = original_output_unflattened[:, transform_indices]
        output_rhs = transformed_output_unflattened.reshape(B, out_feats)

        if not torch.allclose(output_lhs, output_rhs, atol=1e-5):
            print(f"  [!] Test FAILED for solid '{solid_name}', group element h = {h}")
            print(f"      Max difference: {torch.max(torch.abs(output_lhs - output_rhs))}")
            all_tests_passed = False
            break
            
    if all_tests_passed:
        print(f"  [✓] All equivariance tests passed successfully for '{solid_name}'!")

if __name__ == '__main__':
    for solid_name in PLATONIC_GROUPS:
        print(f"\n{'='*25} TESTING: {solid_name.upper()} {'='*25}")
        run_equivariance_test(solid_name)