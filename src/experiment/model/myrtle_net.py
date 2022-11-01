from neural_tangents import stax

from functools import partial

import jax.numpy as jnp

def MyrtleNetwork(N, depth, W_std=jnp.sqrt(2.0), b_std=0.0):
  layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
  width = N
  activation_fn = stax.Relu()
  layers = []
  conv = partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')
  
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][0]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][1]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][2]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 3

  layers += [stax.Flatten(), stax.Dense(10, W_std, b_std)]

  return stax.serial(*layers)