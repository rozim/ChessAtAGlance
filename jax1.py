from typing import Sequence
import sys
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

# Define a simple Linen model
class SimpleLinenModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    print("x=", x, type(x))
    x = nn.Dense(features=64)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

# Create an instance of the Linen model
key = jax.random.PRNGKey(0)
input_shape = (1, 784)  # Example input shape for MNIST
model = SimpleLinenModel()

# Initialize the model parameters
params = model.init(key, jnp.ones(input_shape, jnp.float32))

print(model.tabulate(key, jnp.ones(input_shape, jnp.float32)))



# Print out the model
print(model)
sys.exit(0)


# class MLP(nn.Module):
#   features: Sequence[int]

#   @nn.compact
#   def __call__(self, x):
#     for feat in self.features[:-1]:
#       x = nn.relu(nn.Dense(feat)(x))
#     return x

# model = MLP([12, 8, 4])
# batch = jnp.ones((32, 10))
# variables = model.init(jax.random.key(0), batch)
# output = model.apply(variables, batch)
# #print(output)
# #print()
# print(str(model))
# print('...')
# print(repr(model))
