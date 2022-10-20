from abc import ABC, abstractmethod
import logging
from typing import Mapping
from jax import devices, device_put_replicated, device_put

from save_helpers import create_tmp_folders

class PreprocessDevice(ABC):
    # TODO: does replicate belong here? Fix.
    def __init__(self, save_dir: str, data_params: Mapping, replicate=True):
        self.save_dir = save_dir
        self.data_dir = data_params.root_dir

        try:
            create_tmp_folders(self.data_dir, self.save_dir)
        except OSError:
            logging.error('Could not create temporary folders.')
            raise
        
        _data = self.load_data(data_params)
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


