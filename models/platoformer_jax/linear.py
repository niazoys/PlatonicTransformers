import jax
import jax.numpy as jnp
from jax import Array, random
from flax import linen as nn
import math

from models.platoformer_jax.groups import PLATONIC_GROUPS

class PlatonicLinear(nn.Module):
    """
    A Linear layer constrained to be a group convolution over a Platonic Solid group.
    This version includes a corrected initialization scheme to preserve variance.
    """
    in_features: int
    out_features: int
    solid: str
    use_bias: bool = True
    
    def setup(self):
        if self.solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(f"Solid '{self.solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}")
        
        group = PLATONIC_GROUPS[self.solid.lower()]
        self.G = group.G
        
        if self.in_features % self.G != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by the group order {self.G} for solid '{self.solid}'")
        if self.out_features % self.G != 0:
            raise ValueError(f"out_features ({self.out_features}) must be divisible by the group order {self.G} for solid '{self.solid}'")
            
        self.in_channels = self.in_features // self.G
        self.out_channels = self.out_features // self.G
        
        # Store group properties (these are non-trainable)
        self.cayley_table = group.cayley_table
        self.inverse_indices = group.inverse_indices
        
        # Initialize kernel parameter
        fan_in = self.G * self.in_channels
        std = 1.0 / math.sqrt(fan_in)
        
        self.kernel = self.param(
            'kernel',
            lambda rng: random.normal(rng, (self.G, self.out_channels, self.in_channels)) * std
        )
        
        # Initialize bias parameter if needed
        if self.use_bias:
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            self.bias = self.param(
                'bias',
                lambda rng: random.uniform(rng, (self.out_channels,), minval=-bound, maxval=bound)
            )
    
    def get_weight(self) -> Array:
        """
        Constructs the full [G*O, G*I] weight matrix from the fundamental kernel.
        """
        h_indices = jnp.arange(self.G).reshape(self.G, 1)
        g_indices = jnp.arange(self.G).reshape(1, self.G)
        
        inv_g_indices = self.inverse_indices[g_indices]
        kernel_group_idx = self.cayley_table[h_indices, inv_g_indices]
        
        expanded_kernel = self.kernel[kernel_group_idx]
        weight = expanded_kernel.transpose(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return weight
    
    def __call__(self, x: Array) -> Array:
        weight = self.get_weight()
        output = x @ weight.T
        
        if self.use_bias:
            output_shape = output.shape
            output = output.reshape(*output_shape[:-1], self.G, self.out_channels)
            output = output + self.bias
            output = output.reshape(output_shape)
            
        return output
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(G={self.G}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"

# # Equivariance test
# def run_equivariance_test(solid_name: str, rng_key):
#     group = PLATONIC_GROUPS[solid_name]
#     G = group.G
    
#     I, O, B = 4, 8, 2
#     in_feats = G * I
#     out_feats = G * O

#     print(f"Initializing PlatonicLinear(solid='{solid_name}')")
#     print(f"  (G={G}, I={I}, O={O}) -> in_features={in_feats}, out_features={out_feats}")
    
#     # Initialize model
#     model = PlatonicLinear(in_features=in_feats, out_features=out_feats, solid=solid_name)
    
#     # Split RNG keys
#     init_key, input_key = random.split(rng_key)
    
#     # Create random input
#     input_signal = random.normal(input_key, (B, in_feats))
    
#     # Initialize parameters
#     params = model.init(init_key, input_signal)
    
#     print("--- Testing Right-Equivariance Property ---")
    
#     all_tests_passed = True
#     original_output = model.apply(params, input_signal)

#     for h in range(G):
#         transform_indices = group.cayley_table[h, :]
        
#         if len(jnp.unique(transform_indices)) != G:
#             print(f"[!] Cayley table error at h={h}. Column is not a permutation!")
#             all_tests_passed = False
#             break

#         input_unflattened = input_signal.reshape(B, G, I)
#         transformed_unflattened = input_unflattened[:, transform_indices, :]
#         transformed_input = transformed_unflattened.reshape(B, in_feats)
#         output_lhs = model.apply(params, transformed_input)

#         original_output_unflattened = original_output.reshape(B, G, O)
#         transformed_output_unflattened = original_output_unflattened[:, transform_indices, :]
#         output_rhs = transformed_output_unflattened.reshape(B, out_feats)

#         if not jnp.allclose(output_lhs, output_rhs, atol=1e-5):
#             print(f"  [!] Test FAILED for solid '{solid_name}', group element h = {h}")
#             print(f"      Max difference: {jnp.max(jnp.abs(output_lhs - output_rhs))}")
#             all_tests_passed = False
#             break
            
#     if all_tests_passed:
#         print(f"  [✓] All equivariance tests passed successfully for '{solid_name}'!")
    
#     return all_tests_passed

# if __name__ == '__main__':
#     # Initialize RNG
#     main_key = random.PRNGKey(42)
    
#     for solid_name in PLATONIC_GROUPS:
#         print(f"\n{'='*25} TESTING: {solid_name.upper()} {'='*25}")
#         main_key, test_key = random.split(main_key)
#         run_equivariance_test(solid_name, test_key)