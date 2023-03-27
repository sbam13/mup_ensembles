from dataclasses import dataclass

import os.path
import logging

import jax.numpy as jnp

@dataclass
class Constants:
    CIFAR5M_FOLDER_NAME = '/n/holyscratch01/pehlevan_lab/Lab/sab/cifar5m-data'
    COUNT_PER_PART = 1_000_448


def load_cifar_5m(parts: tuple =(0, 1, 2, 3, 4, 5)):
    '''Loads parts of CIFAR-5M. Adapted from Nikhil's code.'''
    log = logging.getLogger(__name__)
    
    if len(parts) == 0:
        log.info('No parts specified to download.')


    for ind in sorted(parts):
        log.info('-' * 80)
        log.info('Part {i}')
        part_file_loc = os.path.join(Constants.CIFAR5M_FOLDER_NAME, f'part{ind}.npz')
        part_XY = jnp.load(part_file_loc)        
        
        # channels first
        X_ind = jnp.transpose(part_XY['X'], (0, 3, 1, 2))
        Y_ind = part_XY['Y']
        
        log.info('Data type:' + X_ind.dtype)
        log.info('Count: ' + X_ind.shape[0])
        print(ind, flush=True)
        if ind == parts[0]:
            X_all = X_ind
            Y_all = Y_ind
        else:
            X_all = jnp.concatenate((X_all, X_ind))
            Y_all = jnp.concatenate((Y_all, Y_ind))
        log.info('-' * 80)

    assert len(Y_all.shape) == 2
    
        
    X_tr, Y_tr = torch.ByteTensor(X_all[:-40000]), torch.Tensor(Y_all[:-40000]).long()

    if parts[-1] == 5:
        X_te, Y_te = torch.ByteTensor(X_all[-40000:]), torch.Tensor(Y_all[-40000:]).long()
    else:
        part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/cifar-5m/part5.npz')
        X_ind = np.transpose(part_ind['X'], (0, 3, 1, 2))
        Y_ind = part_ind['Y']
        X_te, Y_te = torch.ByteTensor(X_ind[-40000:]), torch.Tensor(Y_ind[-40000:]).long()