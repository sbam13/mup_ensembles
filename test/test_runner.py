from genericpath import exists
import src.run.TaskRunner as tr
from src.tasks.task import Task

from os.path import exists

# def test_run_repeat_task():
#     task = Task('cifar10', 'resnet18')


def test_tcs_yaml(tmp_path):
    t = Task('m', 'd', {'erg': 2.0}, {'werg': False}, None, None, None)
    tr.save_config(str(tmp_path), t)
    assert exists(str(tmp_path / "task_config.yaml"))
    print((tmp_path / "task_config.yaml").read_content())