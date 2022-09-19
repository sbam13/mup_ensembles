from enum import Enum

from dataset.cifar10 import load_cifar_data
from experiment.model.resnet import WideResnet

from src.run.preprocess_device import PreprocessDevice as PD
from src.tasks.read_tasks import TaskReader as TR
from src.tasks.task import Task


class TaskType(Enum):
    TRAIN_NN = 0
    TRAIN_NTK = 1
    COMPUTE_STATISTICS = 2


class PreprocessDevice(PD):
    def _normalize_data(self, data):
        X0, y = data['train']
        X_test0, y_test = data['test']

        X = X0 / 255.0
        X_test = X_test0 / 255.0

        inds = (y<2)
        X_bin = X[inds]
        y_bin = 2*y[inds]-1.0

        inds_te = (y_test<2)
        X_te_bin = X_test[inds_te]
        y_te_bin = 2.0 * y_test[inds_te] -1.0

        return dict(zip(data.keys(), [(X_bin, y_bin), (X_te_bin, y_te_bin)]))
    
    def load_data(self):
        data = load_cifar_data()
        normalized_data = self._normalize_data(data)
        return normalized_data


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def __init__(self):
        super().__init__()
    
    def read_task(self, config):
        hyperparams = config['hyperparams']
                