# _CIFAR_10_LINK = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

import tensorflow_datasets as tfds

import jax.numpy as jnp

def load_cifar_data() -> dict[str, tuple]:
    train_ds, test_ds = tfds.load('cifar10', split=['train','test'],
                                as_supervised = False,
                                batch_size = -1)
    train_images = tfds.as_numpy(train_ds)
    X0 = jnp.array(train_images['image'])
    y = jnp.array(train_images['label'])

    test_images = tfds.as_numpy(test_ds)
    
    X_test0 = jnp.array(test_images['image'] )
    y_test = jnp.array(test_images['label'] )

    return {'train': (X0, y), 'test': (X_test0, y_test)}