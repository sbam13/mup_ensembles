from abc import ABC, abstractmethod
from jax import devices, device_put_replicated, device_put

from os import mkdir
from os.path import join, exists

class PreprocessDevice(ABC):
    # TODO: does replicate belong here? Fix.
    def __init__(self, save_dir, replicate=True):
        if not exists(save_dir):
            mkdir(save_dir)
        self.save_dir = save_dir
        
        _data = self.load_data()
        if replicate is True:
            self.data = self.replicate_data(_data)
        else:
            self.data = device_put(_data, devices()[0])

    @abstractmethod
    def load_data(self, data_params: dict):
        """Return a Pytree with two leaves: the first containing a set of input data and the second 
        containing a set of covariates."""
        pass

    def replicate_data(self, data):
        device_list = devices()
        r_data = device_put_replicated(data, device_list)
        return r_data


