import logging
from src.run.OnlinePreprocessDevice import OnlinePreprocessDevice

from src.tasks.task import Task, Task_ConfigSubset

from os.path import join, exists
from os import mkdir

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

class OnlineTaskRunner:
    def __init__(self, PD: OnlinePreprocessDevice) -> None:
        self.preprocess_device = PD

    def run_serial_task(self, task: Task):
        raise NotImplementedError

    def run_repeat_task(self, task: Task):
        devices = self.preprocess_device.devices
        
        mp, tp = dict(task.model_params), dict(task.training_params)

        # TODO: hacky way to transition to ensembling within GPU
        # iters = len(widths) // num_devices
        # TODO: num_devices doesn't actually have to divide task.repeat!
        key = task.seed
        # iter_keys = split(key, num=iters)
        # del key

        apply = task.apply_callback

        save_folder = join(self.preprocess_device.save_dir, f'task-{task._id}')
        if exists(save_folder):
            raise RuntimeError(f'Save folder for task {task._id} already exists.')
        else:
            mkdir(save_folder)
            save_config(save_folder, task)

        # TODO: this is hacky. separate into grid-search hyperparams
        # recall `apply` is (RNG, data, model_params, training_params) -> result
        # papply = pmap(apply, static_broadcasted_argnums=(2, 3)) # apply (key, data) -> result

        # data = self.preprocess_device.data
        train_data = self.preprocess_device.train_dataset
        val_data = self.preprocess_device.val_data
        
        minibatch_size = tp['minibatch_size']
        num_workers = tp['num_workers']
        train_loader = DataLoader(train_data, minibatch_size, num_workers=num_workers, drop_last=True, persistent_workers=True, shuffle=True)
        val_loader = DataLoader(val_data, minibatch_size, num_workers=1, drop_last=True, persistent_workers=True, shuffle=False)
        # HACKY: do not use in production ^
        # for batch in range(0, iters):
        #     batch_widths = widths[batch*num_devices:(batch + 1)*num_devices]

        #     key = iter_keys[batch]
            # data is replicated across devices, everything else is not
        _ = apply(key, train_loader, val_loader, devices, mp, tp)

        # idx = batch * num_devices
        # for replica, result in enumerate(batch_results):
        #     local_result = device_get(result)
        #     save_result(save_folder, local_result, fname = f'trial_width_{batch_widths[replica]}_result.pkl')


def save_config(dir: str, task: Task):
    FNAME = 'task_config.yaml'
    abs_path_fname = join(dir, FNAME)
    tcs = Task_ConfigSubset(task.model, task.dataset, task.model_params, 
                            task.training_params, tuple(map(int, task.seed)))
    try:
        with open(abs_path_fname, 'x') as f:
            OmegaConf.save(config=tcs, f=abs_path_fname)
    except OSError:
        logging.error('Could not write task config file.')
        raise

# def save_result(dir: str, result: Result, fname):
#     abs_path_fname = join(dir, fname)
#     try:
#         with open(abs_path_fname, 'xb') as f:
#             dump(result, f)
#     except OSError:
#         logging.error(f'Could not write task result file "{fname}" in directory "{dir}".')
#         raise
