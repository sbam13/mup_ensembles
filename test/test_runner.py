import src.run.TaskRunner as tr
from src.tasks.task import Task


def test_run_repeat_task():
    task = Task('cifar10', 'resnet18')