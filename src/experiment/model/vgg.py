# VGG-16 implementation in flax.
from typing import Any, Callable
import flax.linen as nn

from src.experiment.model.common import NTK_Conv, NTK_Dense

from functools import partial

FILTER_SIZE = (3, 3)

ModuleDef = Any

class ConvBlock(nn.Module):
    features: int = 16
    block_size: int = 2
    activation: Callable = nn.relu
    conv_cls: ModuleDef = NTK_Conv
    kernel_init: Callable = nn.initializers.normal(1.0)

    @nn.compact
    def __call__(self, x):
        conv_cls = partial(self.conv_cls, kernel_size=FILTER_SIZE, 
                            use_bias=True, 
                            kernel_init=self.kernel_init)
        for _ in range(self.block_size):
            x = conv_cls(self.features)(x)
            x = self.activation(x)
        return nn.max_pool(x, FILTER_SIZE, (2, 2), padding='SAME') 

class VGG_12(nn.Module):
    N: int = 64
    conv_block: ModuleDef = ConvBlock
    dense_cls: ModuleDef = NTK_Dense
    dense_init: Callable = nn.initializers.normal(1.0)

    @nn.compact
    def __call__(self, x):
        N = self.N
        block_channels: int = (N, 2 * N, 4 * N)
        for bc in block_channels:
            x = self.conv_block(bc)(x)
        x = x.reshape((x.shape[0], -1)) # flatten (dim (4, 4, 4N))
        x = self.dense_cls(16 * N, kernel_init=self.dense_init)(x)
        x = nn.relu(x)
        x = self.dense_cls(8 * N, kernel_init=self.dense_init)(x)
        x = nn.relu(x)
        return self.dense_cls(1, kernel_init=self.dense_init)(x)



        