from abc import abstractmethod, ABC
from typing import List
from .task import Task
from .task import TaskType


class ReadTasks(ABC):
    def read_tasks(self, configs: List):
        id = 0
        for config in configs:
            self.read_experiment()
    
    @abstractmethod
    def read_task(self, config, task_id):
        pass

    @abstractmethod
    def validate_task(self, task: Task):
        pass


class ReadEnsembleCIFARTasks(ReadTasks):
    def read_task(self, config, task_id):
        return super().read_task(config, task_id)
