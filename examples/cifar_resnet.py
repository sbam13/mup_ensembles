from src.run.preprocess_device import PreprocessDevice

import tensorflow_datasets as tfds

class CIFARDevice(PreprocessDevice):
    def prepare_environment():
        return
    
    def load_data():
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'],
                                    as_supervised = False,
                                    batch_size = -1)
        train_images = tfds.as_numpy(train_ds)
        X0 = jnp.array(train_images['image'])
        y = jnp.array(train_images['label'])
        test_images = tfds.as_numpy(test_ds)
        X_test0 = jnp.array(test_images['image'] )
        y_test = jnp.array(test_images['label'] )

        X = X0 / 255.0
        X_test = X_test0 / 255.0

        inds = (y<2)
        X_bin = X[inds]
        y_bin = 2*y[inds]-1.0

        inds_te = (y_test<2)
        X_te_bin = X_test[inds_te]
        y_te_bin = 2.0 * y_test[inds_te] -1.0
