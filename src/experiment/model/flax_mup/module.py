import jax.numpy as jnp
import flax.linen as nn
import typing as T

from functools import partial

class Readout(nn.Module):
    """Wrapper around nn.Dense. Used by Mup to set different learning rate."""
    features: int
    use_bias: bool = True
    kernel_init: T.Callable = nn.initializers.lecun_normal()
    bias_init: T.Callable = nn.initializers.zeros
    param_dtype: T.Any = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        mup_divisor_init = partial(jnp.ones, dtype=self.param_dtype)
        inputs /= self.variable('mup', 'divisor', mup_divisor_init, tuple()).value
        result = nn.Dense(features=self.features, kernel_init=self.kernel_init, bias_init=self.bias_init,
                          use_bias=self.use_bias, param_dtype=self.param_dtype)(inputs)

        return result