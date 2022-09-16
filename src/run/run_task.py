from typing import Callable
from jax.random import PRNGKey, split
from jax import device_count, pmap

from ..tasks.task import Task

def run_parallel_task(task: Task, key: PRNGKey, data, run_experiment: Callable):
    split_keys = split(key, num=task.repeat)
    
    num_devices = device_count()
    # TODO: add experiments % num_devices check


    for batch in range(0, task.repeat, num_devices):
        prun = pmap(run_experiment, axis='param_init', static_broadcasted_argnums=3)
        result = prun(key, task.apply_callback, data, task.hyperparams)
        task.save_callback(result)

    
# TODO: abstract method
def run_experiment(key, model, data, hyperparams):
    pass