import os
import shutil
from enum import Enum
from typing import Mapping
import chex

import jax.numpy as jnp
from jax import vmap, jit
from jax.random import PRNGKey, permutation
from jax.lax import cond

from torch.utils.data import Seq

from src.experiment.dataset.imagenet import load_imagenet_data

from src.experiment.training.online_momentum import apply
# from src.experiment.training.baseline_training import apply as baseline_apply

from src.run.OnlinePreprocessDevice import OnlinePreprocessDevice as OPD
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

class PreprocessDevice(OPD):
    def load_data(self, data_params):
        return load_imagenet_data(constants.IMAGENET_FOLDER, data_params)


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def validate_task(self, task: Task):
        super().validate_task(task)

    def _read_task(self, config: Mapping):
        try:
            key = PRNGKey(config['seed'])
            task = Task(model='resnet18',
                        dataset='imagenet',
                        model_params=OmegaConf.to_container(config['model_params']),
                        training_params=OmegaConf.to_container(config['training_params']),
                        type_=self.task_type,
                        seed=key,
                        apply_callback=Callbacks.APPLY)
        except KeyError:
            raise ValueError('Task not properly configured.')
        return task
