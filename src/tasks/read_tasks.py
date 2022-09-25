from abc import abstractmethod, ABC

from .task import Task
from build_task_graph import order_tasks

from jax import local_device_count
from jax.random import PRNGKey, split


class TaskReader(ABC):
    def __init__(self, setting: dict, config_list: list[dict]):
        self._tasks = self._read_tasks(setting, config_list)
        self._num_devices = local_device_count()

    @property
    def tasks(self):
        return self._tasks
    
    @tasks.setter
    def tasks(self, setting: dict, config_list: list[dict]):
        self._tasks = self._read_tasks(setting, config_list)
    
    @tasks.deleter
    def tasks(self):
        del self._tasks

    # property
    def _read_tasks(self, setting: dict, hyperparam_list: list[dict]):
        seed = setting['seed']
        repeat = setting['repeat']
        key = PRNGKey(seed)
        keys = split(key, len(hyperparam_list))

        task_list = [self._read_task(k, repeat, hp) for k, hp in zip(keys, hyperparam_list)]
        map(self.validate_task, task_list)
        order_tasks(dict(map(lambda t: (t._id, t.dependencies), task_list)))
        return task_list

    @abstractmethod
    def _read_task(self, key: PRNGKey, hyperparams: dict):
        pass

    @abstractmethod
    def validate_task(self, task: Task):
        if task.parallelize and task.repeat % self._num_devices != 0:
            raise ValueError('Number of repeats does not match')
