# Function for framing the FCNN


from flax import linen as nn
import jax.numpy as jnp


# Define a FCNN neural network
class FCNN(nn.Module):
    num_hidden: int
    num_layers: int
    theta_pools: int
    theta_others: int

    def setup(self):
        # Hidden layers
        self.hidden_layers = [
            nn.Dense(
                self.num_hidden,
                kernel_init=nn.initializers.variance_scaling(
                    scale=1.0, mode='fan_avg', distribution='truncated_normal'
                ),
                #bias_init=nn.initializers.constant(0.0),
                bias_init=nn.initializers.normal(0.1)
            )
            for _ in range(self.num_layers)
        ]
        # Output layers
        self.pools_output_layer = nn.Dense(
            self.theta_pools,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_avg', distribution='truncated_normal'
            ),
            #bias_init=nn.initializers.constant(1.0),
            bias_init=nn.initializers.normal(0.1)
        )
        self.others_output_layer = nn.Dense(
            self.theta_others,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_avg', distribution='truncated_normal'
            ),
            # bias_init=nn.initializers.constant(1.0),
            bias_init=nn.initializers.normal(0.1)
        )
        # LayerNorm
        self.layer_norm = nn.LayerNorm()

    def __call__(self, x, verbose=False):
        # Forward pass through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = nn.relu(x)
            if verbose:
                 # Print activation statistics for debugging
                print(
                    f"Layer {i}: activations min={jnp.min(x)}, max={jnp.max(x)}, mean={jnp.mean(x)}"
                )
        x = self.layer_norm(x)
        # Output layers
        theta_pools = self.pools_output_layer(x)
        theta_others = self.others_output_layer(x)
        
        return theta_pools, theta_others
