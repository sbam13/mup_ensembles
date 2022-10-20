from jax.random import split
from jax import local_device_count, pmap, tree_map, device_get
from src.run.PreprocessDevice import PreprocessDevice

from src.tasks.task import Task

from os.path import join


class TaskRunner:
    def __init__(self, PD: PreprocessDevice) -> None:
        self.preprocess_device = PD

    def run_serial_task(self, task: Task):
        key = task.seed
        keys = split(key, num=task.repeat).reshape((task.repeat, 2))
        data = self.preprocess_device.data

        save_folder = join(self.preprocess_device.save_dir, f'task-{task._id}')

        for r in range(task.repeat):            
            result = task.apply_callback(keys[r], data)
            
            repeat_save_folder = join(save_folder, f'trial-{r}')
            task.save_callback(repeat_save_folder, result)

    def run_repeat_task(self, task: Task):
        num_devices = local_device_count()
        iters = task.repeat // num_devices

        key = task.seed
        apply_keys = split(key, num=task.repeat)
        apply_keys = apply_keys.reshape((iters, num_devices, 2)) # each key has shape (2,)

        apply, save = task.apply_callback, task.save_callback

        save_folder = join(self.preprocess_device.save_dir, f'task-{task._id}')

        # TODO: this is hacky. separate into grid-search hyperparams
        papply = pmap(apply, static_broadcasted_argnums=(2, 3)) # apply (key, data) -> result

        data = self.preprocess_device.data
        
        alpha, N = task.hyperparams['alpha'], task.hyperparams['N']
        for batch in range(0, iters):
            result = papply(apply_keys[batch], data, alpha, N)
            local_result = device_get(result)
            
            idx = batch * num_devices
            for replica in range(num_devices):
                replica_result = tree_map(lambda x: x[replica], result)

                repeat_save_folder = join(save_folder, f'trial-{idx + replica}')
                save(repeat_save_folder, replica_result)
            
            del local_result, result