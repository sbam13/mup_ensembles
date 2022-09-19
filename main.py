#!/usr/bin/env python3
from multiprocessing.sharedctypes import Value
from omegaconf import DictConfig
import hydra

from itertools import product
from os import getcwd

from experiment.names import names

# TODO: sys.excepthook

@hydra.main(config_path='conf', config_name='config.yaml')
def main(cfg: DictConfig):
    setting = cfg['setting']
    dataset, model, repeat, seed = setting['dataset'], setting['model'], setting['repeat'], setting['seed']
    try:
        module_ = names[dataset][model]
    except KeyError:
        raise ValueError('Invalid experimental setting.')
    
    reader = module_.TaskReader()

    hyperparams = cfg['hyperparams']
    task_list = reader.read_tasks(hyperparams) # TODO: product this
        
    save_dir = getcwd()
    PD = module_.PreprocessDevice(save_dir)

    



