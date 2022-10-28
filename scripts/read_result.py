from pickle import load

from omegaconf import OmegaConf

from src.run.constants import REMOTE_RESULTS_FOLDER

import os
from os.path import join

import pickle

def read_result(fname):
    with open(fname, 'rb') as f:
        res = load(f)
    return res


def see_lr_losses(FOLDER=REMOTE_RESULTS_FOLDER):
    results_files = os.listdir(FOLDER)
    for rf in results_files:
        if not rf.startswith('results-20221028'):
            continue
        task_folder = join(FOLDER, rf, 'task-0')
        conf = OmegaConf.load('task_config.pkl')
        with open('trial_0_result.pkl', 'rb') as f:
            res = pickle.load(f)
        lr = conf.training_params.eta_0
        print('LR: ', lr)
        print('Train Losses: ', res.train_losses[-5:], '\n')

if __name__ == '__main__':
    see_lr_losses()
