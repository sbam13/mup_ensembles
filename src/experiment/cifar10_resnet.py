import os
import shutil
from enum import Enum
from typing import Mapping

import jax.numpy as jnp
from jax import vmap

from src.experiment.dataset.cifar10 import load_cifar_data
from src.experiment.training.momentum import apply

from src.run.PreprocessDevice import PreprocessDevice as PD
import src.run.constants as constants
from src.tasks.read_tasks import TaskReader as TR
from src.tasks.task import Task

from flax.core.frozen_dict import freeze


class TaskType(Enum):
    TRAIN_NN = 0
    TRAIN_NTK = 1
    # COMPUTE_STATISTICS = 2


class Callbacks(Enum):
    APPLY = apply
    # SAVE = None


# TODO: add to validate_task the check that batch_size divides train and test size

class PreprocessDevice(PD):
    def _normalize_data(self, data: dict):
        X0, y = data['train']
        X_test0, y_test = data['test']

        # normalize
        g = lambda W: W / jnp.sum(W ** 2, dtype=jnp.float32)
        v_g = vmap(g)

        X = v_g(X0)
        X_test = v_g(X_test0)

        # Classes [0 - 4] are 1, classes [5 - 9] are -1
        cifar2 = lambda labels: 2. * ((labels < 5).astype(jnp.float32)) - 1.

        y, y_test = map(cifar2, (y, y_test))

        return dict(zip(data.keys(), [(X, y), (X_test, y_test)]))

    def _copy_data_into_temp(self, SOURCE_FOLDER = constants.CIFAR_FOLDER):
        DEST_FOLDER = os.path.join(self.data_dir, "cifar-10-batches-py")
        shutil.copytree(SOURCE_FOLDER, DEST_FOLDER)

    def load_data(self, data_params):
        self._copy_data_into_temp()

        data = load_cifar_data(data_params)
        normalized_data = self._normalize_data(data)
        
        return freeze(normalized_data)


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def validate_task(self, task: Task):
        super().validate_task(task)

    def _read_task(self, config: Mapping):
        try:
            task = Task(model='resnet18',
                        dataset='cifar10',
                        model_params=config['model_params'],
                        training_params=config['training_params'],
                        type_=self.task_type,
                        seed=config['seed'],
                        repeat=config['repeat'],
                        parallelize=config['parallelize'],
                        # save_callback=Callbacks.save_callback,
                        apply_callback=Callbacks.APPLY)
        except KeyError:
            raise ValueError('Task not properly configured.')
        return task
