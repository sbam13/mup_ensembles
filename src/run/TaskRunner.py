from jax.random import PRNGKey, split
from jax import local_device_count, pmap, tree_map, device_get
from src.run.preprocess_device import PreprocessDevice

from src.tasks.task import Task

from os import mkdir
from os.path import join, exists


class TaskRunner:
    def __init__(self, PD: PreprocessDevice) -> None:
        self.preprocess_device = PD

    @staticmethod
    def _directory_callback(path: str):
        if not exists(path):
            mkdir(path)

    def run_serial_task(self, task: Task):
        key = task.seed
        keys = split(key, num=2 * task.repeat).reshape((task.repeat, 2, 2))
        data = self.preprocess_device.data

        save_folder = join(self.preprocess_device.save_dir, f'task-{task._id}')
        self._directory_callback(save_folder)

        for r in range(task.repeat):
            aux = task.load_callback(keys[r, 0])
            
            result = task.apply_callback(keys[r, 1], data, aux)
            
            repeat_save_folder = join(save_folder, f'trial-{r}')
            self._directory_callback(repeat_save_folder)
            task.save_callback(repeat_save_folder, result)

    def run_repeat_task(self, task: Task):
        num_devices = local_device_count()
        iters = task.repeat // num_devices

        key = task.seed
        keys = split(key, num=2 * task.repeat)
        
        load_keys, apply_keys = keys[:task.repeat], keys[task.repeat:]
        load_keys = load_keys.reshape((iter, num_devices, -1))
        apply_keys = apply_keys.reshape((iter, num_devices, -1))

        load, apply, save = task.load_callback, task.apply_callback, task.save_callback

        save_folder = join(self.preprocess_device.save_dir, f'task-{task._id}')
        self._directory_callback(save_folder)

        pload = pmap(load, axis='param_init') # load (key) -> aux
        papply = pmap(apply, axis='param_init') # apply (key, data, aux) -> result

        data = self.preprocess_device.data
        for batch in range(0, iters):
            aux = pload(load_keys[batch]) 
            result = papply(apply_keys[batch], data, aux)
            local_result = device_get(result)
            
            idx = batch * num_devices
            for replica in range(num_devices):
                replica_result = tree_map(lambda x: x[replica], result)

                repeat_save_folder = join(save_folder, f'trial-{idx + replica}')
                self._directory_callback(repeat_save_folder)
                save(repeat_save_folder, replica_result)
            
            del aux, local_result, result