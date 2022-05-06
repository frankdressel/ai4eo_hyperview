import jax.nn
import jax.numpy as jnp
from flax import linen as nn


class MLPModel(nn.Module):
    output: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = jnp.mean(x, axis=(1,2))
        x = nn.Dense(512)(x)
        x = nn.normalization.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.2)(x, deterministic=not train)
        x = nn.relu(x)

        x = nn.Dense(256)(x)
        x = nn.normalization.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.2)(x, deterministic=not train)
        x = nn.relu(x)

        x = nn.Dense(self.output)(x)
        return x
