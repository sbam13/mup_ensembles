#!/usr/bin/env python3
import logging

from omegaconf import DictConfig

import hydra

from src.experiment.names import names
from src.run.run_tasks import run_tasks

# TODO: timing!

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    # cfg = (setting, hyperparams)
    setting = cfg.setting
    try:
        ds_name, model_name = setting['dataset'], setting['model']
        module_ = names[ds_name, model_name]
    except KeyError:
        raise ValueError('Invalid experimental setting.')

    hp_list = cfg.hyperparams.task_list
    
    reader = module_.TaskReader(hp_list)
    
    save_dir = cfg.save_folder
    
    log.info('Loading data...')
    PD = module_.PreprocessDevice(save_dir, cfg.hyperparams.data_params)
    log.info('...done.')

    log.info('Running tasks...')
    run_tasks(reader.tasks, PD)
    log.info('...all tasks complete.')


if __name__ == '__main__':
    main()


    



