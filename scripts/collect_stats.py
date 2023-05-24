import jax
import jax.numpy as jnp
import numpy as np
import time

import nest_asyncio

import os.path
import os

import optax
from jax import vmap

import pickle

from collections import defaultdict

nest_asyncio.apply()

from flax.training import checkpoints

from functools import reduce

from tqdm import tqdm

import argparse

# -----------------------------------------------------------------------------
# utils

def load_checkpoint(step, fname):
    ens_size = int(fname.split('_')[1])
    ckpt_dir = os.path.join(BASE_DIR, fname)
    step_name = '[[' + ' '.join([str(step)] * ens_size) + ']]'
    ckpt = checkpoints.restore_checkpoint(ckpt_dir, step=step_name, target=None)
    return ckpt

# extract '128' from 'ens_1_width_128_train_state_1684089917.906'
def get_width_from_fname(fname):
    return int(fname.split('_')[3])

# given lists of keys and values, make a dict with keys as keys and values as list of values
def make_dict(keys, values):
    d = {}
    for k, v in zip(keys, values):
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
    return d

# convert the fname
def convert_fname(s):
    spl = s.split('_')
    tup = (spl[1], spl[3], spl[6])
    return 'ens_{}_width_{}_{}'.format(*tup)

def extract_number(s):
    return int(s.split('[[')[1].split(' ')[-1][:-2])

# take the intersection of a collection of sets
def intersect_sets(sets):
    return reduce(lambda a, b: a & b, sets)

def get_wcsl(width_ckpt_map):
    width_ckpt_step_lists = defaultdict(dict)

    for w, dirs in width_ckpt_map.items():
        for d in dirs:
            bdd = os.path.join(BASE_DIR, d)
            width_ckpt_step_lists[w][d] = set(map(extract_number, os.listdir(bdd)))

    wcsl2 = {}
    for w, wcsl in width_ckpt_step_lists.items():
        s = intersect_sets(wcsl.values())
        wcsl2[w] = list(sorted(s))
    return wcsl2

# jax utils
@jax.jit
def get_loss(logits, labels):
    # logit shape: (1024, 1000), label shape (1024,)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

get_ind_losses = vmap(get_loss, in_axes=(0, None))
get_avg_ind_loss = lambda logits, labels: jnp.mean(get_ind_losses(logits, labels), axis=0)
get_ens_loss = lambda logits, labels: get_loss(jnp.mean(logits, axis=0), labels)

@jax.jit
def get_acc(logits, labels):
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == labels)

get_ind_accs = vmap(get_acc, in_axes=(0, None))
get_avg_ind_acc = lambda logits, labels: jnp.mean(get_ind_accs(logits, labels), axis=0)
get_ens_acc = lambda logits, labels: get_acc(jnp.mean(logits, axis=0), labels)


# get logit corresponding to the true label
@jax.jit
def get_true_logits(logits, labels):
    return logits[jnp.arange(logits.shape[0]), labels]

def tree_concat(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)

def collapse_labels_logits(labels, logits):
    logits = jax.tree_map(lambda z: jax.lax.collapse(z, 0, 2), logits)
    logits = tree_concat(logits)
    
    labels = labels[0]
    labels = jax.tree_map(lambda z: jax.lax.collapse(z, 0, 2), labels)

    return ((logits['train'], labels['train']), (logits['val'], labels['val']))

def get_logits_labels(fnames: list, recorded: list):
    tls = {}
    vls = {}
    
    for step in recorded:
        logits = []
        labels = []
        try:
            for fname in fnames:
                sub_ens_d = load_checkpoint(step, fname)
                logits.append({'train': sub_ens_d['train']['logits'], 'val': sub_ens_d['val']['logits']})
                labels.append({'train': sub_ens_d['train']['labels'], 'val': sub_ens_d['val']['labels']})
        except ValueError:
            print(f'step {step}, fname {fname}')
            raise
        
        tls[step], vls[step] = collapse_labels_logits(labels, logits)
        
    return tls, vls

# compute the losses
def get_ens_loss_ot(tls, vls):
    train_losses = {}
    val_losses = {}
    for step in tls.keys():
        train_logits = tls[step][0]
        val_logits = vls[step][0]
        
        train_labels = tls[step][1]
        val_labels = vls[step][1]
        
        train_losses[step] = get_ens_loss(train_logits, train_labels)
        val_losses[step] = get_ens_loss(val_logits, val_labels)
    return train_losses, val_losses

def get_avg_loss_ot(tls, vls):
    train_losses = {}
    val_losses = {}
    for step in tls.keys():
        train_logits = tls[step][0]
        val_logits = vls[step][0]
        
        train_labels = tls[step][1]
        val_labels = vls[step][1]
        
        train_losses[step] = get_avg_ind_loss(train_logits, train_labels)
        val_losses[step] = get_avg_ind_loss(val_logits, val_labels)
    return train_losses, val_losses

def get_ens_acc_ot(tls, vls):
    train_losses = {}
    val_losses = {}
    for step in tls.keys():
        train_logits = tls[step][0]
        val_logits = vls[step][0]
        
        train_labels = tls[step][1]
        val_labels = vls[step][1]
        
        train_losses[step] = get_ens_acc(train_logits, train_labels)
        val_losses[step] = get_ens_acc(val_logits, val_labels)
    return train_losses, val_losses

def get_avg_acc_ot(tls, vls):
    train_losses = {}
    val_losses = {}
    for step in tls.keys():
        train_logits = tls[step][0]
        val_logits = vls[step][0]
        
        train_labels = tls[step][1]
        val_labels = vls[step][1]
        
        train_losses[step] = get_avg_ind_acc(train_logits, train_labels)
        val_losses[step] = get_avg_ind_acc(val_logits, val_labels)
    return train_losses, val_losses


def get_true_logits_ot(vls):
    true_logits = {}

    vgtl = vmap(get_true_logits, in_axes=(0, None))

    for step in vls.keys():
        val_logits = vls[step][0]
        val_labels = vls[step][1]
        
        true_logits[step] = vgtl(val_logits, val_labels)
    return true_logits
    
def main(BASE_DIR):
    # -----------------------------------------------------------------------------
    # get files

    exps_or_ts = os.listdir(BASE_DIR)
    ts = [title for title in exps_or_ts if 'train_state' in title]

    width_model_ckpt_map = make_dict([get_width_from_fname(t) for t in ts], ts)

    width_ckpt_map = {}
    for k, v in sorted(width_model_ckpt_map.items()):
        width_ckpt_map[k] = list(map(convert_fname, v))

    wcsl = get_wcsl(width_ckpt_map)

    # -----------------------------------------------------------------------------
    # process logits

    avg_train_losses = {}
    avg_val_losses = {}

    ens_train_losses = {}
    ens_val_losses = {}

    avg_train_accs = {}
    avg_val_accs = {}

    ens_train_accs = {}
    ens_val_accs = {}

    true_val_logits = {}

    # iterate through wcsl.keys() using tqdm
    for w in tqdm(wcsl.keys()):
        tls, vls = get_logits_labels(width_ckpt_map[w], wcsl[w]) # fix step issue
        
        avg_train_losses[w], avg_val_losses[w] = get_avg_loss_ot(tls, vls)
        ens_train_losses[w], ens_val_losses[w] = get_ens_loss_ot(tls, vls)

        avg_train_accs[w], avg_val_accs[w] = get_avg_acc_ot(tls, vls)
        ens_train_accs[w], ens_val_accs[w] = get_ens_acc_ot(tls, vls)

        true_val_logits[w] = get_true_logits_ot(vls)
        # compute pointwise logit consistency

    # -----------------------------------------------------------------------------
    SAVE_DIR = os.path.join('/n/pehlevan_lab/Users/sab/', BASE_DIR + '_stats')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # pickle the eight dicts above and save them in SAVE_DIR
    with open(os.path.join(SAVE_DIR, 'avg_train_losses.pkl'), 'wb') as f:
        pickle.dump(avg_train_losses, f)

    with open(os.path.join(SAVE_DIR, 'avg_val_losses.pkl'), 'wb') as f:
        pickle.dump(avg_val_losses, f)

    with open(os.path.join(SAVE_DIR, 'ens_train_losses.pkl'), 'wb') as f:
        pickle.dump(ens_train_losses, f)

    with open(os.path.join(SAVE_DIR, 'ens_val_losses.pkl'), 'wb') as f:
        pickle.dump(ens_val_losses, f)

    with open(os.path.join(SAVE_DIR, 'avg_train_accs.pkl'), 'wb') as f:
        pickle.dump(avg_train_accs, f)

    with open(os.path.join(SAVE_DIR, 'avg_val_accs.pkl'), 'wb') as f:
        pickle.dump(avg_val_accs, f)

    with open(os.path.join(SAVE_DIR, 'ens_train_accs.pkl'), 'wb') as f:
        pickle.dump(ens_train_accs, f)

    with open(os.path.join(SAVE_DIR, 'ens_val_accs.pkl'), 'wb') as f:
        pickle.dump(ens_val_accs, f)

    # pickle true_val_logits and save it in SAVE_DIR
    with open(os.path.join(SAVE_DIR, 'true_val_logits.pkl'), 'wb') as f:
        pickle.dump(true_val_logits, f)

    # -----------------------------------------------------------------------------


if __name__ == '__main__':
    # parse BASE_DIR from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    args = parser.parse_args()
    BASE_DIR = args.base_dir

    main(BASE_DIR)






