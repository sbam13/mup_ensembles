# _CIFAR_10_LINK = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

from logging import warning
from multiprocessing.sharedctypes import Value
from typing import Mapping

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose

import jax.numpy as jnp
import jax.random as jr

from jax.numpy import float32, array

def load_cifar_data(data_params: Mapping) -> dict[str, tuple]:
    data_dir = data_params['root_dir']

    training_data = CIFAR10(data_dir, train=True, download=False)
    test_data = CIFAR10(data_dir, train=False, download=False)

    X0 = training_data.data.astype(float32)
    y = array(training_data.targets, dtype=float32)

    rs, ds, P = data_params['random_subset'], data_params['data_seed'], data_params['P']
    (X0, y) = take_subset((X0, y), rs, ds, P)
    
    X_test0 = test_data.data.astype(float32)
    y_test = array(test_data.targets, dtype=float32)

    return {'train': (X0, y), 'test': (X_test0, y_test)}


def take_subset(data: tuple, random_subset: bool, data_seed, P: int):
    X, y = data
    examples = X.shape[0]
    
    if P > examples:
        warning('Dataset size (P) exceeds training dataset size.')
        r, s= divmod(P, examples)
        temp_X, temp_y = jnp.repeat(X, r), jnp.repeat(y, r)
        X_sub, y_sub = jnp.stack([temp_X, X[:s]]), jnp.stack([temp_y, y[0:s]])
    elif P == examples:
        X_sub, y_sub = X, y
    else:
        inds = None
        if random_subset:
            key = jr.PRNGKey(data_seed)
            inds = jr.choice(key, examples, shape=(P,))
        else:
            inds = jnp.arange(P)
        X_sub, y_sub = X[inds], y[inds]
    
    return X_sub, y_sub
