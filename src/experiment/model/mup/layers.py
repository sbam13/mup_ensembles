from typing import Callable, Tuple, Any

import jax.numpy as jnp

import flax.linen as nn

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any

class MuReadout(nn.Dense):
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(1.0)

	@nn.compact
	def __call__(self, x):
		multiplier = self.param('multiplier', 
								lambda key, shape, dtype: jnp.array(1 / shape[0], dtype=dtype), 
								(jnp.shape(x)[-1], self.features), self.dtype)
		y = super().__call__(x)
		return multiplier * y
