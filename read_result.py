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
        if not rf.startswith('results-20221028'):
            continue
        for task in range(2):
            task_folder = join(FOLDER, rf, f'task-{task}')
            try:
                conf = OmegaConf.load(join(task_folder, 'task_config.pkl'))
            except:
                conf = OmegaConf.load(join(task_folder, 'task_config.yaml'))
            with open(join(task_folder, 'trial_0_result.pkl'), 'rb') as f:
                res = pickle.load(f)
            lr = conf.training_params.eta_0
            print('LR: ', lr)
            print('Train Losses: ', res.train_losses[-5:], '\n')

def see_lr_deviations(FOLDER=REMOTE_RESULTS_FOLDER):
    results_files = os.listdir(FOLDER)
    total_res = []
    for rf in results_files:
        if not rf.startswith('results-20221028'):
            continue
        for task in range(2):
            task_folder = join(FOLDER, rf, f'task-{task}')
            try:
                conf = OmegaConf.load(join(task_folder, 'task_config.pkl'))
            except:
                conf = OmegaConf.load(join(task_folder, 'task_config.yaml'))
            with open(join(task_folder, 'trial_0_result.pkl'), 'rb') as f:
                res = pickle.load(f)
            total_res.append((conf, res))
    return total_res

if __name__ == '__main__':
    see_lr_losses()
