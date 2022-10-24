from functools import partial
import time

from src.tasks.task import Task
from src.run.PreprocessDevice import PreprocessDevice
from src.run.TaskRunner import TaskRunner

from logging import info, error

def _run_task(task: Task, runner: TaskRunner):
    start = time.time()
    try:
        if task.parallelize:
            runner.run_repeat_task(task)
        else:
            runner.run_serial_task(task)
    except BaseException as e:
            error(f'Task {task._id} raised an exception. Task and error specification: ', task.model_params, task.training_params, e) 
    end = time.time()
    elapsed = end - start
    info(f'Task completed. Elapsed time (s): {elapsed}. Task specifications: ', task.model_params, task.training_params)

def run_tasks(tasks: list[Task], specs: PreprocessDevice):
    runner = TaskRunner(specs)

    partial_run_task = partial(_run_task, runner=runner)
    map(partial_run_task, tasks)