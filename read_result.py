from omegaconf import OmegaConf

from src.run.constants import REMOTE_RESULTS_FOLDER

import os
from os.path import join

import pickle

def read_result(fname):
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    return res


def see_lr_losses(FOLDER=REMOTE_RESULTS_FOLDER):
    results_files = os.listdir(FOLDER)
    for rf in results_files:
        if not rf.startswith('results-20221101'):
            continue
        for task in range(1):
            task_folder = join(FOLDER, rf, f'task-{task}')
            try:
                conf = OmegaConf.load(join(task_folder, 'task_config.pkl'))
            except:
                conf = OmegaConf.load(join(task_folder, 'task_config.yaml'))
            with open(join(task_folder, 'trial_0_result.pkl'), 'rb') as f:
                res = pickle.load(f)
            lr = conf.training_params.eta_0
            alpha = conf.model_params.alpha
            print('rf', rf, 'LR: ', lr, ' alpha: ', alpha)
            print('Train Losses: ', res.train_losses[-5:], '\n')

def see_lr_deviations(FOLDER=REMOTE_RESULTS_FOLDER):
    results_files = os.listdir(FOLDER)
    total_res = []
    for rf in results_files:
        res = {}
        rf_loc = join(FOLDER, rf)
        data_conf = OmegaConf.load(join(rf_loc, 'data_config.yaml'))
        res['data_config'] = data_conf
        num_tasks = len(os.listdir(rf_loc)) - 1
        for task in range(num_tasks):
            task_str = f'task-{task}'
            task_folder = join(rf_loc, task_str)
            task_conf = OmegaConf.load(join(task_folder, 'task_config.yaml'))
            num_trials = len(os.listdir(task_folder)) - 1
            trial_results = []
            for trial in range(num_trials):
                with open(join(task_folder, f'trial_{trial}_result.pkl'), 'rb') as f:
                    trial_results.append(pickle.load(f))
            res[task_str] = (task_conf, trial_results)
        total_res.append(res)
    return total_res

if __name__ == '__main__':
    see_lr_losses()
