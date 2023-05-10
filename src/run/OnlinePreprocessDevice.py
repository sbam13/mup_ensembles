from abc import ABC, abstractmethod
from logging import getLogger
from typing import Mapping
import jax
import jax.numpy as jnp

from torch.utils.data import DataLoader
import torch as ch

# from src.run.save_helpers import create_tmp_folder
from os.path import join

from src.run import constants

from omegaconf import OmegaConf

logging = getLogger(__name__)

class OnlinePreprocessDevice(ABC):
    # TODO: does replicate belong here? Fix.
    def __init__(self, base_dir: str, data_params: dict, replicate=True):
        self.save_dir = join(base_dir, constants.LOCAL_RESULTS_FOLDER)
        self.data_params = dict(data_params)
        self.data_dir = join(base_dir, self.data_params['root_dir'])
        
        self.devices = None
        self.train_dataset = None
        
        self.val_data = None

    def _prep_vd(self, vd):
        len_vd = len(vd)
        assert len_vd < 5_000

        NUM_WORKERS = 1
        # vd_batch_size = len_vd // NUM_WORKERS
        vd_loader = DataLoader(vd, num_workers=NUM_WORKERS, batch_size=len_vd, shuffle=False, drop_last=False)

        self.val_data = next(iter(vd_loader))

    def preprocess(self, parallelize=True):
        """Initializes the PreprocessDevice object."""
        # assume validation dataset is small enough to put in memory
        
        self._save_data_params(self.save_dir, self.data_params)

        td, vd = self.load_data(self.data_params)

        self.train_dataset = td

        logging.info('Reading validation data:')
        self._prep_vd(vd)
        logging.info('Done reading validation data.')
        # logging.info(f'Put validation data onto device.')
        # else: # no loading onto device
        #     self.val_data = (jnp_images, jnp_labels)


    @abstractmethod
    def load_data(self, data_params: Mapping):
        pass

    def _save_data_params(self, dir:str, data_params: dict):
        fname = 'data_config.yaml'
        abs_path_fname = join(dir, fname)
        dp_yaml = OmegaConf.to_yaml(data_params)
        try:
            with open(abs_path_fname, 'x') as f:
                f.write(dp_yaml)
        except OSError as e:
            logging.error('Could not write data config file.', e)
            raise
        



