import itertools
import os
import os.path
import re
import time
from functools import partial
from logging import info
from typing import Any

import chex
import flax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import checkpoints, train_state
from jax import ShapeDtypeStruct, jit, tree_map, value_and_grad, vmap
from jax.lax import scan
from jax.random import split

from src.experiment.model.flax_mup.resnet import ResNet18
from src.experiment.model.flax_mup.mup import Mup

from src.run.constants import BASE_SAVE_DIR

NUM_CLASSES = 1_000
IMAGENET_SHAPE = (224, 224, 3)

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def initialize(keys, N: int, num_ensemble_subsets: int, mup, param_dtype):
    model = ResNet18(num_classes=1000, num_filters=N, param_dtype=param_dtype)
    dummy_input = jnp.zeros((1,) + IMAGENET_SHAPE, dtype=param_dtype) # added batch index

    wp = mup._width_mults
    rzi = mup.readout_zero_init
    
    def train_init(key, inp):
        vars_ = model.init(key, inp, train=True)
        mup_vars = dict(mup.rescale_parameters({'params': vars_['params']}, wp, rzi), **{'batch_stats': vars_['batch_stats']})
        mup_vars.update({'mup': vars_['mup']})
        return mup_vars
    
    within_subset_size = keys.shape[0] // num_ensemble_subsets
    sub_keys = keys.reshape((num_ensemble_subsets, within_subset_size, 2))
    
    ensemble_get_params = vmap(vmap(train_init, in_axes=(0, None), axis_name='within_subset'), in_axes=(0, None), axis_name='over_subsets')
    fn = jit(ensemble_get_params).lower(ShapeDtypeStruct(sub_keys.shape, jnp.uint32), ShapeDtypeStruct(dummy_input.shape, jnp.float32)).compile()
    ws = fn(sub_keys, dummy_input)
    return ws


def train(vars_0: chex.ArrayTree, N: int, optimizer: optax.GradientTransformation, 
        train_loader, val_data, epochs: int, batch_size: int = 64, 
        n_ensemble: int = 32, ensemble_subsets: int = 1, use_checkpoint: bool = False, ckpt_dir: str = '', model_ckpt_dir: str = '',
        data_dtype: Any = jnp.float32) -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    """`vars_0` has shape (n_ensemble, param_dims...) for every leaf value"""
    tranche_size = train_loader.batch_size
    num_batches = tranche_size // batch_size # 0 is sharding dimension

    class TrainState(train_state.TrainState):
        batch_stats: chex.ArrayTree
        mup: chex.ArrayTree

    @partial(vmap, in_axes=(0, None, None), axis_name='ensemble')
    def _subset_update(state: TrainState, Xtr_sb: chex.ArrayDevice, ytr_sb: chex.ArrayDevice) -> TrainState:
        """Runs one minibatch (formerly epoch)."""
        def apply_fn(vars, Xin):
            """Returns a tuple (y_hat, updated_batch_stats)."""
            y_hat, dict_updated_bs = state.apply_fn(vars, Xin, train=True, mutable=['batch_stats'])
            return y_hat, dict_updated_bs['batch_stats']
        
        def loss_fn(params, batch_stats, mup_col, Xin, yin):
            vars = {'params': params, 'batch_stats': batch_stats, 'mup': mup_col}
            y_hat, update_bs = apply_fn(vars, Xin)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_hat, yin))
            return loss, update_bs
        
        loss_grad_fn = value_and_grad(loss_fn, has_aux=True)
        # -----------------------------------------------------------

        def step(step_state: TrainState, data: tuple) -> TrainState:
            """Takes an SGD step."""
            # unpack
            batch, labels = data

            # update params
            (_, update_bs), grads = loss_grad_fn(step_state.params, step_state.batch_stats, step_state.mup, batch, labels)
            return step_state.apply_gradients(grads=grads, batch_stats=update_bs), None
    

        # shuffle
        # Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data

        # SGD steps over batches in minibatch
        updated_state, _ = scan(step, state, (Xtr_sb, ytr_sb))

        return updated_state

    @jit
    def update(state: TrainState, Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice):
        Xtr_sb = Xtr.reshape((num_batches, batch_size, *Xtr.shape[1:]))
        ytr_sb = ytr.reshape((num_batches, batch_size, *ytr.shape[1:]))

        def partial_subset_update(state_stacked):
            return _subset_update(state_stacked, Xtr_sb, ytr_sb)
        
        return jax.lax.map(partial_subset_update, state)

    # -------------------------------------------------------------------------
    # data collection helper
    def get_preds(state, x, y):
        num_batches = x.shape[0] // batch_size
        x_sb = x.reshape((num_batches, batch_size, *x.shape[1:]))
        y_sb = y.reshape((num_batches, batch_size, *y.shape[1:]))

        @partial(vmap, axis_name='within_subset', in_axes=0)
        def get_subset_preds(ind_state):
            variables = {'params': ind_state.params, 'batch_stats': ind_state.batch_stats, 'mup': ind_state.mup}
            def inference_subroutine(batch_data):
                x_batch, _ = batch_data
                logits = state.apply_fn(variables, x_batch, train=False)
                
                # logits_max = jnp.max(logits, axis=-1, keepdims=True)
                # logits -= jax.lax.stop_gradient(logits_max)
                
                return logits

            batched_logits = jax.lax.map(inference_subroutine, (x_sb, y_sb))
            return jax.lax.collapse(batched_logits, 0, 2) # collapse data batch dimension

        all_logits = jax.lax.map(get_subset_preds, state)
        return {'logits': all_logits, 'labels': y}


    get_layer_norms = lambda ind_state: tree_map(lambda z: jnp.linalg.norm(z), ind_state.params)
    # -------------------------------------------------------------------------
    model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, param_dtype=data_dtype)

    def create_train_state(params, tx, bs, mup):
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=bs, mup=mup)

    # removed pmap
    init_params = vars_0['params']
    init_bs = vars_0['batch_stats']
    mup_col= vars_0['mup']
    init_step_state = vmap(vmap(create_train_state, axis_name='within_subset', in_axes=(0, None, 0, 0)), 
                        axis_name='over_subsets', in_axes=(0, None, 0, 0))(init_params, optimizer, init_bs, mup_col)

    # training loop
    state = init_step_state
    glp = jit(get_preds).lower(init_step_state, jax.ShapeDtypeStruct((tranche_size, 224, 224, 3), data_dtype), jax.ShapeDtypeStruct((tranche_size,), jnp.dtype('int32'))).compile()
    gln = jit(vmap(vmap(get_layer_norms)))
    # -------------------------------------------------------------------------
    # create checkpointer
    info('create checkpointer')
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    time_suffix = time.time()

    def save_stats(state, train_x, train_y, val_x, val_y):
        train_preds = glp(state, train_x, train_y)
        val_preds = glp(state, val_x, val_y)
        layer_norms = gln(state)
        ckpt = {'train': train_preds, 'val': val_preds, 'layer_norms': layer_norms}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=state.step, overwrite=False, keep=1_000_000,
                                orbax_checkpointer=checkpointer)
        return

    def cleanup_save_stats(state, val_x, val_y):
        val_preds = glp(state, val_x, val_y)
        ckpt = {'val': val_preds}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=state.step, overwrite=False, keep=1_000_000,
                                orbax_checkpointer=checkpointer)
        return
    info('...done')
    # -------------------------------------------------------------------------
    # helper to convert dataloader batches to jax
    def loader_to_jax(batch):
        x_ch, y_list = batch
        x_jnp = jnp.array(x_ch, dtype=data_dtype)
        y_jnp = jnp.array(y_list)
        return x_jnp, y_jnp
    # -------------------------------------------------------------------------
    jax_val_data = loader_to_jax(val_data)
    # -------------------------------------------------------------------------
    # if use_checkpoint:
    #     model_ckpt_fname = os.listdir(model_ckpt_dir)[-1]
    #     prev_record_step = steps = int(re.findall(r'\d+', model_ckpt_fname)[0])

    #     info('restore model checkpoint...')
    #     # fix model_ckpt_dir to be passed in
    #     state = checkpoints.restore_checkpoint(model_ckpt_dir, state, orbax_checkpointer=checkpointer)
        
    #     # hacky, replace in future trials
    #     vectorized_steps = steps * jnp.ones((ensemble_subsets, n_ensemble // ensemble_subsets), dtype=jnp.int32)
    #     state = state.replace(step=vectorized_steps)
    #     info('...restored!')

    #     # skip seen batches to align across experiments
    #     batches_seen = steps // num_batches # num_batches is minibatch_size // microbatch_size
    #     total_batches = len(train_loader)
    #     skip_batches = batches_seen % total_batches
        
    #     info(f'skip {skip_batches} batches to align across experiments...')
    #     for _ in itertools.islice(data_iter, skip_batches):
    #         pass
    #     info('...done!')
    # else:
    ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}_{time_suffix:.3f}')
    model_ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}_train_state_{time_suffix:.3f}')

    # -------------------------------------------------------------------------
    def exp_scale(start, stop, exponent):
        ret = [start]
        while ret[-1] < stop:
            ret.append(int(np.ceil(ret[-1] * exponent)))
        return frozenset([0] + ret[:-1])

    LOG_SPACING_MULTIPLIER = 1.15
    LOG_SCALE_SAVE_THRESHOLD = 102_400

    tranche_save_threshold = LOG_SCALE_SAVE_THRESHOLD // tranche_size


    checkpoint_save_tranches = exp_scale(1, tranche_save_threshold, LOG_SPACING_MULTIPLIER)

    LINEAR_SPACING = 50

    def should_save_checkpoint(tranches_seen):
        if tranches_seen < tranche_save_threshold:
            return tranches_seen in checkpoint_save_tranches
        else:
            return (tranches_seen - tranche_save_threshold) % LINEAR_SPACING == 0
    # -------------------------------------------------------------------------
    tranches_seen = 0


    info('Entering training loop...')
    start = time.time()

    for _ in range(epochs):
        for tranche in map(loader_to_jax, iter(train_loader)):
            x, y = tranche
            if should_save_checkpoint(tranches_seen):
                save_stats(state, x, y, *jax_val_data)
                info(f'images {tranches_seen * tranche_size}: elapsed time {time.time() - start}')
                if tranches_seen >= tranche_save_threshold:
                    checkpoints.save_checkpoint(model_ckpt_dir, state, step=state.step, orbax_checkpointer=checkpointer, keep=1_000_000)
            state = update(state, x, y)
            tranches_seen += 1

    info(f'...exiting loop: elapsed time {time.time() - start}')
    # note that return value is a pytree
    return None, None


# def get_n_ensemble(width):
#     NUM_ENS_64 = 64
#     _exp = np.log2(width) - 6.0
#     ens = NUM_ENS_64 // (4 ** _exp)
#     return int(ens)


# def get_div_size(N: int, ensemble_size: int):
#     div = min(int((2 ** 10) // N), ensemble_size)
#     if N == 4:
#         return div // 4
#     if N in [16, 128]:
#         return div // 2
#     else:
#         return div

def apply(key, train_loader, val_data, devices, model_params, training_params):
    # get sharded keys
    # n_ensemble = model_params['repeat']
    n_ensemble = model_params['ensemble_size']
    assert n_ensemble > 0

    ensemble_subsets = training_params['ensemble_subsets']
    assert ensemble_subsets > 0 and n_ensemble % ensemble_subsets == 0

    BASE_N = model_params['BASE_N']
    N = model_params['N']
    assert N > 0

    try:
        dtype = jnp.dtype(model_params['dtype'])
    except TypeError:
        raise ValueError('`model_params.param_dtype` must be a valid jax dtype')

    # -------------------------------------------------------------------------
    # set up muP

    mup = Mup()

    init_input = jnp.zeros((1,) + IMAGENET_SHAPE, dtype=dtype)
    base_model = ResNet18(num_classes=NUM_CLASSES, num_filters=BASE_N, param_dtype=dtype)
    vars_ = base_model.init(jax.random.PRNGKey(0), init_input)
    mup.set_base_shapes({'params': vars_['params']})

    target_model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, param_dtype=dtype)
    vars_target = target_model.init(jax.random.PRNGKey(0), init_input)
    mup.set_target_shapes({'params': vars_target['params']})
    del vars_, vars_target, base_model, target_model, init_input
    # -------------------------------------------------------------------------

    # initialize!
    # key, subsample_key = split(key)
    init_keys = split(key, num=n_ensemble)
    del key

    # alpha = model_params['alpha']
    vars_0 = initialize(init_keys, N, ensemble_subsets, mup, dtype) # shape: (n_ensemble // div, div, param dims)
    # NUM_VMAP_DIMENSIONS = 2
    info('initialized parameters!')

    # create optimizer ----------------------------------------------------------------------
    eta_0 = training_params['eta_0']

    # warmup and cosine decay schedule
    base_optimizer = None
    use_warmup_cosine_decay = training_params['use_warmup_cosine_decay']
    if use_warmup_cosine_decay:
        wcd_params = training_params['wcd_params']
        warmup_epochs = wcd_params['warmup_epochs']
        init_lr = wcd_params['init_lr']
        min_lr = wcd_params['min_lr']

        steps_per_minibatch = training_params['minibatch_size'] // training_params['microbatch_size']
        steps_per_epoch = steps_per_minibatch * len(train_loader)
        
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        decay_steps = training_params['epochs'] * steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=init_lr, peak_value=eta_0, warmup_steps=warmup_steps, decay_steps=decay_steps, end_value=min_lr)
        base_optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        base_optimizer = optax.adam(eta_0)

    optimizer = mup.wrap_optimizer(base_optimizer, adam=True)

    batch_size = training_params['microbatch_size']
    epochs = training_params['epochs']

    # checkpoint data
    use_checkpoint = training_params['use_checkpoint']
    ckpt_dir = training_params['ckpt_dir']
    model_ckpt_dir = training_params['model_ckpt_dir']

    # train!
    info('entering train function')
    _ = train(vars_0, 
            N,
            optimizer, 
            train_loader, 
            val_data, 
            epochs, 
            batch_size, 
            n_ensemble,
            ensemble_subsets,
            use_checkpoint,
            ckpt_dir,
            model_ckpt_dir,
            dtype)

    return None

