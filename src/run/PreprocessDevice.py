from abc import ABC, abstractmethod
import logging
from typing import Mapping
import jax

from src.run.save_helpers import create_tmp_folder
from os.path import join

from omegaconf import OmegaConf

class PreprocessDevice(ABC):
    # TODO: does replicate belong here? Fix.
    def __init__(self, save_dir: str, data_params: dict, replicate=True):
        self.save_dir = save_dir
        self.data_params = dict(data_params)
        self.data_dir = self.data_params['root_dir']
        
        self.devices = None
        self.data = None

    def preprocess(self):
        """Initializes the PreprocessDevice object."""
        try:
            create_tmp_folder(self.save_dir)
            create_tmp_folder(self.data_dir)
        except OSError:
            logging.error('Could not create temporary folders.')
            raise
        
        self._save_data_params(self.save_dir, self.data_params)

        _data = self.load_data(self.data_params)

        # maintains pmap order of devices
        self.devices = jax.lib.xla_bridge.get_backend().get_default_device_assignment(jax.device_count())
        if replicate:
            self.data = jax.device_put_replicated(_data, self.devices)
            logging.info('Replicated data onto devices.')
        else: # no loading onto device
            self.data = _data


    @abstractmethod
    def load_data(self, data_params: Mapping):
        """Return a Pytree with two leaves: the first containing a set of input data and the second 
        containing a set of covariates."""
        pass

    def _save_data_params(self, dir:str, data_params: dict):
        fname = 'data_config.yaml'
        abs_path_fname = join(dir, fname)
        dp_yaml = OmegaConf.to_yaml(data_params)
        try:
            with open(abs_path_fname, 'x') as f:
                f.write(dp_yaml)
        except OSError:
            logging.error('Could not write task config file.')
            raise
        



