from cmath import sqrt
import logging
import chex
import jax.numpy as jnp

from optax._src import base

def blocked_polynomial_schedule(
    init_value: chex.Scalar,
    power: chex.Scalar,
    transition_begin: int = 0,
    block_steps: int = 4
) -> base.Schedule:
  if transition_begin < 0:
    logging.info(
        'An exponential schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.')
    transition_begin = 0

  block_steps = float(block_steps)
  def schedule(count):
    count = jnp.clip(count - transition_begin, a_min=0)
    count_level = jnp.floor(count / block_steps)
    lr = init_value * ((1 + count_level) ** power)
    return lr
  return schedule
