from enum import Enum

from experiment.dataset.cifar10 import load_cifar_data
from experiment.model.resnet import WideResnet

from src.run.preprocess_device import PreprocessDevice as PD
from src.tasks.read_tasks import TaskReader as TR
from src.tasks.task import Task

from jax.random import PRNGKey
from jax.numpy import float32

class TaskType(Enum):
    TRAIN_NN = 0
    TRAIN_NTK = 1
    COMPUTE_STATISTICS = 2


def _load_callback(P: int):
    pass


# TODO: add to validate_task the check that batch_size divides train and test size

class PreprocessDevice(PD):
    def _normalize_data(self, data):
        X0, y = data['train']
        X_test0, y_test = data['test']

        X = X0 / 255.0
        X_test = X_test0 / 255.0

        # Classes [0 - 4] are 1, classes [5 - 9] are -1
        cifar2 = lambda labels: 2. * ((labels < 5).astype(float32)) - 1.

        y, y_test = map(cifar2, (y, y_test))

        return dict(zip(data.keys(), [(X_bin, y_bin), (X_te_bin, y_te_bin)]))
    
    def load_data(self, data_params):
        P = data_params['P']
        data = load_cifar_data()
        
        normalized_data = self._normalize_data(data)
        return normalized_data


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def __init__(self):
        super().__init__()

    def validate_config(self, config: dict):
        model, data, training = config['model_params'], config['data_params'], config['training_params']
        
        batch_size
        
    
    def _read_task(self, key, repeat, hyperparams):
        Task(hyperparams=hyperparams, type_=self.task_type, seed=key, repeat=repeat, parallelize=True)

                