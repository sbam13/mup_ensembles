from typing import Callable, Tuple, Any

import jax.numpy as jnp

import flax.linen as nn

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


class MuReadout(nn.Dense):
	alpha: float = 1.0
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(1.0)
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(1.0)

	@nn.compact
	def __call__(self, x):
		multiplier = self.param('multiplier', 
								lambda _, shape, dtype: jnp.array(self.alpha / shape[0], dtype=dtype), 
								(jnp.shape(x)[-1], self.features), self.dtype)
		return super().__call__(multiplier * x)
