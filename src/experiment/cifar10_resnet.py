import os
import shutil
from enum import Enum
from typing import Mapping

import jax.numpy as jnp
from jax import vmap, jit
from jax.random import PRNGKey
from jax.lax import cond

from src.experiment.dataset.cifar10 import load_cifar_data
from src.experiment.training.jaxopt_momentum import apply
# from src.experiment.training.stax_momentum import apply as stax_apply
# from src.experiment.training.baseline_training import apply as baseline_apply

from src.run.PreprocessDevice import PreprocessDevice as PD
import src.run.constants as constants
from src.tasks.read_tasks import TaskReader as TR
from src.tasks.task import Task

from flax.core.frozen_dict import freeze

from omegaconf import OmegaConf

class TaskType(Enum):
    TRAIN_NN = 0
    TRAIN_NTK = 1
    # COMPUTE_STATISTICS = 2


class Callbacks(Enum):
    APPLY = apply
    # APPLY = baseline_apply
    # SAVE = None


# TODO: add to validate_task the check that batch_size divides train and test size

class PreprocessDevice(PD):
    def _whiten_data(self, data: dict):
        """Center and normalize so that x-values are on the sphere."""
        X0, y = data['train']
        X_test0, y_test = data['test']

        # ---------------------------------------------------------------------
        # X0 = self._preprocess_im2gray(X0)
        # X_test0 = self._preprocess_im2gray(X_test0)
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # center using training mean
        train_mean = jnp.mean(X0, axis=0)
        X0 -= train_mean
        X_test0 -= train_mean

        # scale down by stddev
        def g(z):
            return jnp.sqrt(jnp.sum(z ** 2))
        train_scale = jnp.mean(jit(vmap(g))(X0))
        X0 /= train_scale
        X_test0 /= train_scale
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # SCALING_CONSTANT = 255.0
        # X0 /= SCALING_CONSTANT
        # X_test0 /= SCALING_CONSTANT
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # normalize
        # def normalize(W): 
        #     # TODO: make this jax.lax.cond
        #     im_norm = jnp.sum(W ** 2, dtype=jnp.float32)
        #     div_by_scalar = lambda z, c: z / c 
        #     id = lambda z, c: z
        #     return cond(jnp.isclose(im_norm, 0.0), id, div_by_scalar, W, jnp.sqrt(im_norm))

        # v_normalize = jit(vmap(normalize))

        # X0 = v_normalize(X0)
        # X_test0 = v_normalize(X_test0)
        # ---------------------------------------------------------------------


        # Classes [0 - 4] are 1, classes [5 - 9] are -1
        cifar2 = lambda labels: 2. * ((labels < 5).astype(jnp.float32)) - 1.

        y, y_test = map(cifar2, (y, y_test))

        return dict(train=(X0, y), test=(X_test0, y_test))

    def _preprocess_im2gray(self, X):
        SCALING_CONSTANT = 255.0
        
        def rgb2gray(rgb):
            r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        return jnp.expand_dims(rgb2gray(X) / SCALING_CONSTANT, axis=3)


    def _copy_data_into_temp(self, SOURCE_FOLDER = constants.CIFAR_FOLDER):
        DEST_FOLDER = os.path.join(self.data_dir, "cifar-10-batches-py")
        shutil.copytree(SOURCE_FOLDER, DEST_FOLDER)

    def load_data(self, data_params):
        # self._copy_data_into_temp()

        data = load_cifar_data(self.data_dir, data_params)
        whitened_data = self._whiten_data(data)
        
        return freeze(whitened_data)


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def validate_task(self, task: Task):
        super().validate_task(task)

    def _read_task(self, config: Mapping):
        try:
            key = PRNGKey(config['seed'])
            task = Task(model='resnet18',
                        dataset='cifar10',
                        model_params=OmegaConf.to_container(config['model_params']),
                        training_params=OmegaConf.to_container(config['training_params']),
                        type_=self.task_type,
                        seed=key,
                        repeat=config['repeat'],
                        parallelize=config['parallelize'],
                        # save_callback=Callbacks.save_callback,
                        apply_callback=Callbacks.APPLY)
        except KeyError:
            raise ValueError('Task not properly configured.')
        return task
