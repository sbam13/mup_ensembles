#!/usr/bin/env python3
from multiprocessing.sharedctypes import Value
from omegaconf import DictConfig
import hydra

from itertools import product
from os import getcwd

from experiment.names import names
from src.run.TaskRunner import TaskRunner

from logging import info
from time import time

# TODO: sys.excepthook, add global finally that moves scratch 
# TODO: change savedir to scratch

@hydra.main(config_path='conf', config_name='config.yaml')
def main(cfg: DictConfig):
    setting = cfg['setting']
    try:
        module_ = names[setting['dataset']][setting['model']]
    except KeyError:
        raise ValueError('Invalid experimental setting.')

    hyperparams = cfg['hyperparams']
    hp_list = []
    for vt in product(hyperparams.values()):
        hp_list.append(dict(zip(hyperparams.keys(), vt)))
    
    reader = module_.TaskReader(setting, hp_list)
    
    save_dir = getcwd()
    PD = module_.PreprocessDevice(save_dir)

    runner = TaskRunner(PD)

    for idx, task in enumerate(reader.tasks):
        start = time.time()
        if task.parallelize:
            runner.run_repeat_task(task)
        else:
            runner.run_serial_task(task)
        end = time.time()
        elapsed = end - start
        info(f'Task {idx} completed. Elapsed time (s): {elapsed}.')
        


    



