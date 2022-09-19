import jax.numpy as jnp

from neural_tangents import stax
from functools import partial

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3), strides,W_std=jnp.sqrt(2.0), padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3), W_std=jnp.sqrt(2.0), padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides,W_std=jnp.sqrt(2.0), padding='SAME')
  return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1)):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1))]
  return stax.serial(*blocks)

def WideResnet(block_size, k, num_classes):
  return stax.serial(
      stax.Conv(16*k, (3, 3), W_std = jnp.sqrt(2.0), padding='SAME'),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      stax.AvgPool((8, 8)),
      stax.Flatten(),
      stax.Dense(num_classes, W_std=jnp.sqrt(2.0)))


def MyrtleNetwork(depth, width, W_std=np.sqrt(2.0), b_std=1.0, num_classes = 1):
  layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
  width = 1
  activation_fn = stax.Relu()
  layers = []
  conv = partial(stax.Conv, W_std=W_std,b_std=b_std, padding='SAME')

  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][0]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][1]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][2]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]

  layers += [stax.Flatten(), stax.Dense(num_classes, W_std, b_std=b_std)]

  return stax.serial(*layers)