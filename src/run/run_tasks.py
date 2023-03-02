from functools import partial
import time

from src.tasks.task import Task
from src.run.PreprocessDevice import PreprocessDevice
from src.run.OnlineTaskRunner import OnlineTaskRunner

from logging import info, error

def _run_task(task: Task, runner: OnlineTaskRunner):
    info(f'Task {task._id} starting...')
    start = time.time()
    try:
        if task.parallelize:
            runner.run_repeat_task(task)
        else:
            runner.run_serial_task(task)
    except BaseException as e:
            error(f'Task {task._id} raised an exception.') 
            raise
    end = time.time()
    elapsed = end - start
    info(f'Task {task._id} completed. Elapsed time (s): {elapsed}.')

def run_tasks(tasks: list[Task], specs: PreprocessDevice):
    runner = OnlineTaskRunner(specs)

    for task in tasks:
        _run_task(task, runner)
