"""Flax implementation of WideResnet from neural_tangent.stax."""

from functools import partial
from typing import Any, Callable

from flax import linen as nn

from src.experiment.model import common

ModuleDef = Any

class WideResnetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
#   norm: ModuleDef
  strides: tuple[int, int] = (1, 1)
  act: Callable = nn.relu
  channel_mismatch: bool = False

  @nn.compact
  def __call__(self, x):
    if not self.channel_mismatch:
      res = x
    else:
      res = self.conv(self.filters, (3, 3), strides=self.strides, padding='SAME')(x)

    x1 = self.act(x)
    x2 = self.conv(self.filters, (3, 3), strides=self.strides, padding='SAME')(x1)
    
    x3 = self.act(x2)
    x4 = self.conv(self.filters, (3, 3), padding='SAME')(x3)
    
    return res + x4


class WideResnetGroup(nn.Module):
  layers: int
  filters: int
  conv_cls: ModuleDef
  strides: tuple[int, int] = (1, 1)
  block_cls: ModuleDef = WideResnetBlock

  @nn.compact
  def __call__(self, x):
    x = self.block_cls(self.filters, self.conv_cls, self.strides, channel_mismatch=True)(x)
    for _ in range(self.layers - 1):
      x = self.block_cls(self.filters, self.conv_cls)(x)
    return x


class WideResnet(nn.Module):
  block_size: int
  k: int
  num_classes: int
  conv_cls: ModuleDef = common.NTK_Conv
  dense_cls: ModuleDef = common.NTK_Dense
  group_cls: ModuleDef = WideResnetGroup
  conv_init: Callable = nn.initializers.normal(1.0)
  dense_init: Callable = nn.initializers.normal(1.0)

  @nn.compact
  def __call__(self, x):
    conv_cls = partial(self.conv_cls, kernel_init=self.conv_init)
    
    x = conv_cls(16, (3, 3), padding='SAME')(x)
    x = self.group_cls(self.block_size, 16 * self.k, conv_cls)(x)
    x = self.group_cls(self.block_size, 32 * self.k, conv_cls, (2, 2))(x)
    x = self.group_cls(self.block_size, 64 * self.k, conv_cls, (2, 2))(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    return self.dense_cls(self.num_classes, kernel_init=self.dense_init)(x)
  


class MF_WideResnet(nn.Module):
  block_size: int
  k: int
  num_classes: int
  conv_cls: ModuleDef = common.MF_Conv
  dense_cls: ModuleDef = common.MF_Dense
  group_cls: ModuleDef = WideResnetGroup
  conv_init: Callable = nn.initializers.normal(1.0)
  dense_init: Callable = nn.initializers.normal(1.0)

  @nn.compact
  def __call__(self, x):
    conv_cls = partial(self.conv_cls, kernel_init=self.conv_init)
    
    x = conv_cls(16, (3, 3), padding='SAME')(x)
    x = self.group_cls(self.block_size, 16 * self.k, conv_cls)(x)
    x = self.group_cls(self.block_size, 32 * self.k, conv_cls, (2, 2))(x)
    x = self.group_cls(self.block_size, 64 * self.k, conv_cls, (2, 2))(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    return self.dense_cls(self.num_classes, kernel_init=self.dense_init)(x)