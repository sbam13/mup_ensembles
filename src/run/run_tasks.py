from functools import partial

from src.tasks.task import Task
from src.run.preprocess_device import PreprocessDevice
from src.run.TaskRunner import TaskRunner

def _run_task(task: Task, runner: TaskRunner):
    if task.parallelize:
        runner.run_repeat_task(task)
    else:
        runner.run_serial_task(task)

def run_tasks(tasks: list[Task], specs: PreprocessDevice):
    runner = TaskRunner(specs)

    partial_run_task = partial(_run_task, runner=runner)
    map(partial_run_task, tasks)