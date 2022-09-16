from abc import ABC, abstractmethod
from jax import devices, device_put_replicated

import jax.numpy as jnp
import tensorflow_datasets as tfds # FIX


class PreprocessDevice(ABC):
    @abstractmethod
    def prepare_environment():
        """Modify the runtime environment."""
        pass

    @abstractmethod
    def load_data():
        """Return a Pytree with two leaves: the first containing a set of input data and the second 
        containing a set of covariates."""
        pass

    def replicate_data(data):
        device_list = devices()
        r_data = device_put_replicated(data, device_list)
        return r_data


