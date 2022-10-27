from itertools import product
from numpy import int32, asarray
from omegaconf import OmegaConf
from os.path import join, dirname

import jax.random as jr

from config_structs import Config, DataParams, ModelParams, Setting, TaskConfig, TaskListConfig

CONFIG_DIR = '../conf/experiment'
SAVE_DIR = '/tmp/results'

def gen_sweeps(alpha_vals, N_vals, P_vals, ensemble_size: int, 
            bagging_size: int, seed: int, data_seed: int):
    k = jr.PRNGKey(seed)
    lP, la, lN = len(P_vals), len(alpha_vals), len(N_vals)
    seeds = asarray(jr.randint(k, (lP * la * lN,), 0, 10**6
                                ).reshape((lP, la, lN)))

    data_key = jr.PRNGKey(data_seed)
    data_seeds = asarray(jr.randint(data_key, (bagging_size,), 0, 10**6))

    curr_dir = dirname(__file__)
    config_save_folder = join(curr_dir, CONFIG_DIR)

    for bag in range(len(data_seeds)):
        s_D = data_seeds[bag]
        for i in range(len(P_vals)):
            P = P_vals[i]
            seed_matrix = seeds[i]
            config_str = _gen_sweep(alpha_vals, N_vals, P, bag, 
                        es=ensemble_size, seed_matrix=seed_matrix, data_seed=s_D)
            config_fname = f'sweep_dataset_size_{P}_data_bag_{bag}.yaml'
            config_rel_loc = join(config_save_folder, config_fname)

            with open(config_rel_loc, mode='x') as fi:
                fi.write(config_str) # TODO: add file exists exception handler + clean up



def _gen_sweep(alpha_vals, N_vals, P, bag, es, seed_matrix, data_seed):
    dp = DataParams(P=P, data_seed=int(data_seed))
    tasks = TaskListConfig(data_params=dp)
    
    for j in range(len(N_vals)):
        N = N_vals[j]
        for i in range(len(alpha_vals)):
            a = alpha_vals[i]
            seed = seed_matrix[i, j]
            
            mp = ModelParams(N, a)
            a_N_task = TaskConfig(model_params=mp, repeat=es, seed=int(seed))
            tasks.task_list.append(a_N_task)
    
    setting = Setting()
    save_folder = join(SAVE_DIR, f'results_dataset_size_{P}_data_seed_{bag}')
    conf = Config(setting, tasks, save_folder)

    str_conf = OmegaConf.to_yaml(conf)
    return '# @package _global_\n' + str_conf


if __name__ == '__main__':
    gen_sweeps([0.5], [64], [16384], 4, 1, 7473, 2289)