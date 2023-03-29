from numpy import int32, asarray
from omegaconf import OmegaConf
from os.path import join, dirname
import os, shutil, re

from template import SBATCH_TEMPLATE
from math import ceil
from config_structs import Config, DataParams, ModelParams, TrainingParams, Setting, TaskConfig, TaskListConfig

CONFIG_DIR = '../conf/experiment'
SBATCH_DIR = '../sbatch_files'

BASE_DIR = '/tmp/{id}'

CONFIG_NAME = 'sweep_width_{id}.yaml'
SBATCH_NAME = 'sweep_width_{id}.bat'

BASE_CKPT_DIR = '/n/pehlevan_lab/Users/sab/ensemble_compute_data/'


# def gen_sweeps(mo_vals, lr_vals, alpha_vals, N_vals, P_vals, ensemble_size: int, ngpus: int,
#             bagging_size: int, seed: int, data_seed: int):
#     k = jr.PRNGKey(seed)
#     lP, la, lN = len(P_vals), len(alpha_vals), len(N_vals)
#     seeds = asarray(jr.randint(k, (lP * la * lN,), 0, 10**6
#                                 ).reshape((lP, la, lN)))

#     data_key = jr.PRNGKey(data_seed)
#     data_seeds = asarray(jr.randint(data_key, (bagging_size,), 0, 10**6))

#     curr_dir = dirname(__file__)
#     config_save_folder = join(curr_dir, CONFIG_DIR)
#     sbatch_save_folder = join(curr_dir, SBATCH_DIR)

#     id = 0
#     for mo in mo_vals:
#         for lr in lr_vals:
#             for bag in range(len(data_seeds)):
#                 s_D = data_seeds[bag]
#                 for i in range(len(P_vals)):
#                     P = P_vals[i]
#                     seed_matrix = seeds[i]
#                     config_str = _gen_sweep(id, lr, mo, alpha_vals, N_vals, P, 
#                                 es=ensemble_size, seed_matrix=seed_matrix, data_seed=s_D)
#                     config_fname = CONFIG_NAME.format(id=id)
#                     config_rel_loc = join(config_save_folder, config_fname)

#                     sbatch_str = SBATCH_TEMPLATE.format(id=id, ngpus=ngpus)

#                     sbatch_fname = SBATCH_NAME.format(id=id)
#                     sbatch_rel_loc = join(sbatch_save_folder, sbatch_fname)



#                     with open(config_rel_loc, mode='x') as fi:
#                         fi.write(config_str) # TODO: add file exists exception handler + clean up
#                     with open(sbatch_rel_loc, mode='x') as fi:
#                         fi.write(sbatch_str) # TODO: add file exists exception handler + clean up
#                     id += 1


# def _gen_sweep(id, lr, mo, alpha_vals, N_vals, P, es, seed_matrix, data_seed):
#     dp = DataParams(P=P, data_seed=int(data_seed))
#     tasks = TaskListConfig(data_params=dp)
#     STEPS = 24000
#     BATCH_SIZE = 256
#     num_epochs = ceil(STEPS / (P / BATCH_SIZE))
#     for j in range(len(N_vals)):
#         N = N_vals[j]
#         for i in range(len(alpha_vals)):
#             a = alpha_vals[i]
#             seed = seed_matrix[i, j]
#             tp = TrainingParams(eta_0=lr, epochs=num_epochs, batch_size=BATCH_SIZE, momentum=mo)
#             mp = ModelParams(N, a)
#             a_N_task = TaskConfig(model_params=mp, training_params=tp, repeat=es, seed=int(seed))
#             tasks.task_list.append(a_N_task)
    
#     setting = Setting()
#     conf = Config(setting, tasks, BASE_DIR.format(id=id))

#     str_conf = OmegaConf.to_yaml(conf)
#     return '# @package _global_\n' + str_conf


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_ckpt(model_ckpt):
    numbers = re.findall(r'\d+\.\d+|\d+', model_ckpt)
    # the regular expression '\d+\.\d+|\d+' matches either a decimal number or an integer

    tuple_numbers = tuple(float(num) if '.' in num else int(num) for num in numbers)
    return 'ens_{:0}_width_{:1}_{:2}'.format(*tuple_numbers)

def get_tp(model_ckpt_dir):
    ckpt_dir = get_ckpt(model_ckpt_dir)
    return TrainingParams(use_checkpoint=True, ckpt_dir=os.path.join(BASE_CKPT_DIR, ckpt_dir), model_ckpt_dir=os.path.join(BASE_CKPT_DIR, model_ckpt_dir))

if __name__ == '__main__':
    clear_folder(CONFIG_DIR)
    clear_folder(SBATCH_DIR)

    data_seed = 2423
    seed = 3442
    dp = DataParams(data_seed=data_seed)

    width_es_map =  {512: 1, 256: 2, 128: 6, 64: 12, 32: 36, 16: 64, 8: 64, 4: 128, 2: 128}

    width_model_ckpt_map = {256:  ['ens_2_width_256_train_state_1679953979.468',
                                'ens_2_width_256_train_state_1679953987.165',
                                'ens_2_width_256_train_state_1679954017.611',
                                'ens_2_width_256_train_state_1679954029.819',
                                'ens_2_width_256_train_state_1679954032.606',
                                'ens_2_width_256_train_state_1679954057.219',
                                'ens_2_width_256_train_state_1679954134.250',
                                'ens_2_width_256_train_state_1679954179.080',
                                'ens_2_width_256_train_state_1679954202.039',
                                'ens_2_width_256_train_state_1679972304.076',
                                'ens_2_width_256_train_state_1679972381.922',
                                'ens_2_width_256_train_state_1679986566.213',
                                'ens_2_width_256_train_state_1679992949.629',
                                'ens_2_width_256_train_state_1680004755.925'],
                            128: ['ens_6_width_128_train_state_1679953900.619',
                                'ens_6_width_128_train_state_1679953939.994',
                                'ens_6_width_128_train_state_1679953948.630',
                                'ens_6_width_128_train_state_1679953952.762',
                                'ens_6_width_128_train_state_1679953969.855',
                                'ens_6_width_128_train_state_1679953986.463'],
                            64: ['ens_12_width_64_train_state_1679953833.194',
                                'ens_12_width_64_train_state_1679953846.059'],
                            512: ['ens_1_width_512_train_state_1679955124.003',
                                'ens_1_width_512_train_state_1679955570.377',
                                'ens_1_width_512_train_state_1679960228.979',
                                'ens_1_width_512_train_state_1679960254.207',
                                'ens_1_width_512_train_state_1679960320.385',
                                'ens_1_width_512_train_state_1679960725.670',
                                'ens_1_width_512_train_state_1679960876.808',
                                'ens_1_width_512_train_state_1679960974.454'],
                            32: ['ens_36_width_32_train_state_1679953824.461'],
                            16: ['ens_64_width_16_train_state_1679953821.214'],
                            8: ['ens_64_width_8_train_state_1679953816.044'],
                            4: ['ens_128_width_4_train_state_1679953822.498']}

    tlc_list = []
    for k, v in width_model_ckpt_map.items():
        mp = ModelParams(N=k, ensemble_size=width_es_map[k])
        tlc_list += [TaskListConfig(task_list=[TaskConfig(training_params=tp, model_params=mp, seed=seed)], data_params=dp) for tp in map(get_tp, v)]

    configs = [Config(setting=Setting(), hyperparams=tlc_inst, base_dir=BASE_DIR.format(id=id)) for id, tlc_inst in enumerate(tlc_list)]
    str_configs = ['# @package _global_\n' + OmegaConf.to_yaml(conf) for conf in configs]

    curr_dir = dirname(__file__)
    config_save_folder = join(curr_dir, CONFIG_DIR)
    sbatch_save_folder = join(curr_dir, SBATCH_DIR)

    for id, strc in enumerate(str_configs):
        config_fname = CONFIG_NAME.format(id=id)
        config_rel_loc = join(config_save_folder, config_fname)

        sbatch_str = SBATCH_TEMPLATE.format(id=id)

        sbatch_fname = SBATCH_NAME.format(id=id)
        sbatch_rel_loc = join(sbatch_save_folder, sbatch_fname)

        with open(config_rel_loc, mode='x') as fi:
            fi.write(strc) # TODO: add file exists exception handler + clean up
        with open(sbatch_rel_loc, mode='x') as fi:
            fi.write(sbatch_str) # TODO: add file exists exception handler + clean up

