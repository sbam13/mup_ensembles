from functools import partial
from logging import info

import chex
import jax.numpy as jnp
import optax
import flax

from flax.training import train_state, checkpoints

from jax import jit, vmap, tree_map, value_and_grad, ShapeDtypeStruct
from jax.lax import scan

import jax.lax
import orbax

import os.path
import os

import time, re

from jax.random import split
# from src.experiment.training.Result import OnlineResult

from src.experiment.model.mup import ResNet18

from src.experiment.training.prefetch import prefetch_to_device

import numpy as np

from functools import partial

NUM_CLASSES = 1_000
PREFETCH_N = 2
SPACING_MULTIPLIER = 1.3
BASE_SAVE_DIR = '/n/pehlevan_lab/Users/sab/ensemble_compute_data'

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def initialize(keys, N: int, alpha: float, num_ensemble_subsets: int):
    model = ResNet18(num_classes=1000, num_filters=N, alpha=alpha)
    IMAGENET_SHAPE = (224, 224, 3)
    dummy_input = jnp.zeros((1,) + IMAGENET_SHAPE) # added batch index
    
    train_init = partial(model.init, train=True)
    
    within_subset_size = keys.shape[0] // num_ensemble_subsets
    sub_keys = keys.reshape((num_ensemble_subsets, within_subset_size, 2))
    
    ensemble_get_params = vmap(vmap(train_init, in_axes=(0, None), axis_name='within_subset'), in_axes=(0, None), axis_name='over_subsets')
    fn = jit(ensemble_get_params).lower(ShapeDtypeStruct(sub_keys.shape, jnp.uint32), ShapeDtypeStruct(dummy_input.shape, jnp.float32)).compile()
    ws = fn(sub_keys, dummy_input)
    return {'params': ws['params'].unfreeze(), 'batch_stats': ws['batch_stats']}


def train(vars_0: chex.ArrayTree, N: int, alpha: float, optimizer: optax.GradientTransformation, 
        train_loader, X_val, y_val, minibatches: int, batch_size: int = 64, 
        n_ensemble: int = 32, ensemble_subsets: int = 1, use_checkpoint: bool = False, ckpt_dir: str = '', model_ckpt_dir: str = '') -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    """`vars_0` has shape (n_ensemble, param_dims...) for every leaf value"""
    tranche_size = train_loader.batch_size
    num_batches = tranche_size // batch_size # 0 is sharding dimension

    class TrainState(train_state.TrainState):
        batch_stats: chex.ArrayTree

    @partial(vmap, in_axes=(0, None, None), axis_name='ensemble')
    def _subset_update(state: TrainState, Xtr_sb: chex.ArrayDevice, ytr_sb: chex.ArrayDevice) -> TrainState:
        """Runs one minibatch (formerly epoch)."""
        def apply_fn(vars, Xin):
            """Returns a tuple (y_hat, updated_batch_stats)."""
            y_hat, dict_updated_bs = state.apply_fn(vars, Xin, train=True, mutable=['batch_stats'])
            return y_hat, dict_updated_bs['batch_stats']
        
        def loss_fn(params, batch_stats, Xin, yin):
            vars = {'params': params, 'batch_stats': batch_stats}
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
            (_, update_bs), grads = loss_grad_fn(step_state.params, step_state.batch_stats, batch, labels)
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
            variables = {'params': ind_state.params, 'batch_stats': ind_state.batch_stats}
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
    model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, alpha=alpha)

    def create_train_state(params, tx, bs):
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=bs)

    # removed pmap
    init_params = vars_0['params']
    init_bs = vars_0['batch_stats']
    init_step_state = vmap(vmap(create_train_state, axis_name='within_subset', in_axes=(0, None, 0)), 
                        axis_name='over_subsets', in_axes=(0, None, 0))(init_params, optimizer, init_bs)

    # training loop
    state = init_step_state
    glp = jit(get_preds).lower(init_step_state, jax.ShapeDtypeStruct((tranche_size, 224, 224, 3), jnp.dtype('float32')), jax.ShapeDtypeStruct((tranche_size,), jnp.dtype('int32'))).compile()
    gln = jit(vmap(vmap(get_layer_norms)))
    # -------------------------------------------------------------------------
    # create checkpointer
    info('create checkpointer')
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    time_suffix = time.time()

    def save_stats(step, state, train_x, train_y, val_x, val_y):
        train_preds = glp(state, train_x, train_y)
        val_preds = glp(state, val_x, val_y)
        layer_norms = gln(state)
        ckpt = {'train': train_preds, 'val': val_preds, 'layer_norms': layer_norms}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=step, overwrite=False, keep=1000,
                                orbax_checkpointer=checkpointer)
        return

    def cleanup_save_stats(step, state, val_x, val_y):
        val_preds = glp(state, val_x, val_y)
        ckpt = {'val': val_preds}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=step, overwrite=False, keep=1000,
                                orbax_checkpointer=checkpointer)
        return
    info('...done')
    # -------------------------------------------------------------------------

    steps = 0
    prev_record_step = 0

    if use_checkpoint:
        model_ckpt_fname = os.listdir(model_ckpt_dir)[-1]
        steps = int(re.findall(r'\d+', model_ckpt_fname)[0])

        info('restore model checkpoint...')
        # fix model_ckpt_dir to be passed in
        state = checkpoints.restore_checkpoint(model_ckpt_dir, state, orbax_checkpointer=checkpointer)
        
        # hacky, replace in future trials
        vectorized_steps = steps * jnp.ones((ensemble_subsets, n_ensemble // ensemble_subsets), dtype=jnp.int32)
        state = state.replace(step=vectorized_steps)
        info('...restored!')
    else:
        ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}_{time_suffix:.3f}')
        model_ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}_train_state_{time_suffix:.3f}')


    info('Entering training loop...')
    try:
        start = time.time()

        prefetch_iter = prefetch_to_device(train_loader, PREFETCH_N)
        for _ in range(minibatches):
            batch = next(prefetch_iter)
            x, y = batch
            state = update(state, x, y)
            steps += num_batches
            if steps > SPACING_MULTIPLIER * prev_record_step:
                prev_record_step = steps
                save_stats(steps, state, x, y, X_val, y_val)
                info(f'step {steps}: elapsed time {time.time() - start}')
                if steps > 7_500: # 480_000 data points seen
                    checkpoints.save_checkpoint(model_ckpt_dir, state, step=steps, orbax_checkpointer=checkpointer)
    finally:
        cleanup_save_stats(steps, state, X_val, y_val)
        checkpoints.save_checkpoint(model_ckpt_dir, state, step=steps, orbax_checkpointer=checkpointer)

    info('...exiting loop.')
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

    N = model_params['N']
    assert N > 0

    # initialize!
    # key, subsample_key = split(key)
    init_keys = split(key, num=n_ensemble)
    del key

    alpha = model_params['alpha']
    vars_0 = initialize(init_keys, N, alpha, ensemble_subsets) # shape: (n_ensemble // div, div, param dims)
    # NUM_VMAP_DIMENSIONS = 2
    info('initialized parameters!')

    # create optimizer ----------------------------------------------------------------------
    eta_0 = training_params['eta_0']

    def flattened_traversal(fn):
        """Returns function that is called with `(path, param)` instead of pytree."""
        def mask(tree):
            flat = flax.traverse_util.flatten_dict(tree)
            return flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat.items()})
        return mask

    def assign_lr(k, v):
        layer_name = k[-2]
        if layer_name.startswith('Conv'):
            return eta_0 / np.prod(v.shape[:-1])
        else:
            return eta_0

    lr_fn = flattened_traversal(assign_lr)
    lrs = lr_fn(vars_0['params'])
    
    opts = jax.tree_map(lambda lr: optax.adam(lr), lrs)
    flat_opts = flax.traverse_util.flatten_dict(opts)
    
    # fix multiplier
    flat_opts[('MuReadout_0', 'multiplier')] = optax.set_to_zero()

    # scale readout learning rates by 1/alpha
    flat_opts[('MuReadout_0', 'kernel')] = optax.adam(eta_0 / alpha)
    flat_opts[('MuReadout_0', 'bias')] = optax.adam(eta_0 / alpha)

    str_flat_opts = {str.join(' -- ', k): v for k, v in flat_opts.items()}

    label_mapping = flattened_traversal(lambda k, _: str.join(' -- ', k))(vars_0['params'])

    optimizer = optax.multi_transform(str_flat_opts, label_mapping)
    # ---------------------------------------------------------------------------------------

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
            alpha,
            optimizer, 
            train_loader, 
            *val_data, 
            epochs * 1024, 
            batch_size, 
            n_ensemble,
            ensemble_subsets,
            use_checkpoint,
            ckpt_dir,
            model_ckpt_dir)

    return None

