from abc import abstractmethod, ABC

from .task import Task
from build_task_graph import order_tasks

from jax import local_device_count


class TaskReader(ABC):
    def __init__(self, configs: list[dict]):
        self.tasks = self.read_tasks(configs)

    # property
    def _read_tasks(self, configs: list):
        task_list = [self.read_task(config) for config in configs]
        map(self.validate_task, task_list)
        order_tasks(dict(map(lambda t: (t._id, t.dependencies), task_list)))
        return task_list

    @abstractmethod
    def _read_task(self, config: dict):
        pass

    @abstractmethod
    def validate_task(self, task: Task):
        num_devices = local_device_count()
        if task.parallelize and task.repeat % num_devices != 0:
            raise ValueError('Number of repeats does not match')
