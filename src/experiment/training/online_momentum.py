from functools import partial
from logging import info

import chex
import jax.numpy as jnp
import optax

from flax.training import train_state, checkpoints

from jax import jit, vmap, tree_map, value_and_grad, ShapeDtypeStruct
from jax.lax import scan

import jax.lax
import orbax

import os.path
import os

import time

from jax.random import split
# from src.experiment.training.Result import OnlineResult

from src.experiment.model.mup import ResNet18

from src.experiment.training.prefetch import prefetch_to_device

import numpy as np

from functools import partial

NUM_CLASSES = 1_000
PREFETCH_N = 2
SPACING_MULTIPLIER = 1.4
BASE_SAVE_DIR = '/n/pehlevan_lab/Users/sab/ensemble_compute_data'

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def initialize(keys, N: int, div: int):
    model = ResNet18(num_classes=1000, num_filters=N)
    IMAGENET_SHAPE = (224, 224, 3)
    dummy_input = jnp.zeros((1,) + IMAGENET_SHAPE) # added batch index
    
    # def get_params(key, dummy):
    #     w_frozen = model.init(key, dummy)
    #     return w_frozen.unfreeze()
    train_init = partial(model.init, train=True)
    
    subsets = keys.shape[0] // div
    sub_keys = keys.reshape((subsets, div, 2))
    
    ensemble_get_params = vmap(vmap(train_init, in_axes=(0, None), axis_name='within_subset'), in_axes=(0, None), axis_name='over_subsets')
    fn = jit(ensemble_get_params).lower(ShapeDtypeStruct(sub_keys.shape, jnp.uint32), ShapeDtypeStruct(dummy_input.shape, jnp.float32)).compile()
    return fn(sub_keys, dummy_input)


def train(vars_0: chex.ArrayTree, N: int, optimizer: optax.GradientTransformation, 
        train_loader, X_val, y_val, epochs: int = 10, batch_size: int = 64, 
        n_ensemble: int = 32) -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    """`vars_0` has shape (n_ensemble, param_dims...) for every leaf value"""
    tranche_size = train_loader.batch_size
    num_batches = tranche_size // batch_size # 0 is sharding dimension

    class TrainState(train_state.TrainState):
        batch_stats: chex.ArrayTree

    # # @partial(pmap, axis_name='width')
    # @partial(jit, static_argnums=(0, 4))
    # def compute_average_ensemble_loss(key: chex.PRNGKey, state: TrainState, X: chex.ArrayDevice, y: chex.ArrayDevice, samples_per_ensemble: int = 50):
    #     """Returns a tuple consisting of (1) the average loss for each network in an ensemble and (2) the loss of the ensemble."""
    #     @vmap(axis_name='within_subset', in_axes=0)
    #     def get_subset_individual_yhat_loss(ind_state):
    #         variables = {'params': ind_state.params, 'batch_stats': ind_state.batch_stats}
    #         yhat = state.apply_fn(variables, X, train=False)
    #         individual_loss = cross_entropy(yhat, y)
    #         return yhat, individual_loss
        
    #     def get_individual_yhat_loss(state):
    #         jax.lax.map(get_subset_individual_yhat_loss, state)


    #     yhats, losses = get_individual_yhat_loss(state)

    #     def compute_ensemble_loss(ensemble_yhats, y_true):
    #         ensemble_preds = jnp.mean(ensemble_yhats, axis=0)
    #         return cross_entropy(ensemble_preds, y_true)
        
    #     @vmap
    #     def sample_average_ensemble_loss(idx):
    #         return jnp.mean(losses[idx]), compute_ensemble_loss(yhats[idx], y)

    #     losses = {}
    #     size = n_ensemble // 2
    #     while size > 1:
    #         key, other = split(key)
    #         keys = split(other, num=samples_per_ensemble)
    #         sample_idx = vmap(choice, in_axes=(0, None, None))(keys, n_ensemble, (size,))
    #         avg_loss, ens_loss = sample_average_ensemble_loss(sample_idx)
    #         losses[size] = {'avg': avg_loss, 'ens': ens_loss}

    #     losses[1] = {'avg': losses, 'ens': losses}
    #     losses[n_ensemble] = {'avg': jnp.mean(losses), 'ens': compute_ensemble_loss(yhats, y)}
    #     return losses

    
    @jit
    @vmap(in_axes=(0, None, None), axis_name='ensemble')
    def _subset_update(state: TrainState, Xtr_sb: chex.ArrayDevice, ytr_sb: chex.ArrayDevice) -> TrainState:
        """Runs one minibatch (formerly epoch)."""
        def apply_fn(vars, Xin):
            """Returns a tuple (y_hat, updated_batch_stats)."""
            y_hat, dict_updated_bs = state.apply_fn(vars, Xin, train=True, mutable=['batch_stats'])
            return y_hat, dict_updated_bs['batch_stats']
        
        def loss_fn(params, batch_stats, Xin, yin):
            vars = {'params': params, 'batch_stats': batch_stats}
            y_hat, update_bs = apply_fn(vars, Xin)
            loss = optax.softmax_cross_entropy_with_integer_labels(y_hat, yin)
            return loss, (y_hat, update_bs)
        
        loss_grad_fn = value_and_grad(loss_fn, has_aux=True)
        # -----------------------------------------------------------

        def step(step_state: TrainState, data: tuple) -> TrainState:
            """Takes an SGD step."""
            # unpack
            opt_state = step_state.opt_state
            batch, labels = data

            # update params
            # param_shape = tree_map(lambda z: z.shape, params)
            # data_shape = tree_map(lambda z: z.shape, batch)
            (_, (_, update_bs)), grads = loss_grad_fn(step_state.params, step_state.batch_stats, batch, labels)
            updates, opt_state = optimizer.update(grads, opt_state, step_state.params)
            
            # apply updates only to mutable params
            updated_params = optax.apply_updates(step_state.params, updates)

            return step_state.replace(params=updated_params, batch_stats=update_bs, opt_state=opt_state), None
    

        # shuffle
        # Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data

        # SGD steps over batches in minibatch
        updated_state, _ = scan(step, state, (Xtr_sb, ytr_sb))

        return updated_state

    def update(state: TrainState, Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice):
        Xtr_sb = Xtr.reshape((num_batches, batch_size, *Xtr.shape[1:]))
        ytr_sb = ytr.reshape((num_batches, batch_size, *ytr.shape[1:]))

        def partial_subset_update(state_stacked):
            return _subset_update(state_stacked, Xtr_sb, ytr_sb)

        return jax.lax.map(partial_subset_update, state)

    # -------------------------------------------------------------------------
    # data collection helper
    def get_loss_stats(state, x, y):
        num_batches = x.shape[0] // batch_size
        x_sb = x.reshape((num_batches, batch_size, *x.shape[1:]))
        y_sb = y.reshape((num_batches, batch_size, *y.shape[1:]))

        @partial(vmap, axis_name='within_subset', in_axes=0)
        def get_subset_loss_stats(ind_state):
            variables = {'params': ind_state.params, 'batch_stats': ind_state.batch_stats}
            def loss_stats_subroutine(batch_data):
                x_batch, batch_targets = batch_data
                logits = state.apply_fn(variables, x_batch, train=False)
                
                logits_max = jnp.max(logits, axis=-1, keepdims=True)
                logits -= jax.lax.stop_gradient(logits_max)
                
                label_logits = jnp.take_along_axis(logits, batch_targets[..., None], axis=-1)[..., 0]
                log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
                ens_log_normalizers = jnp.log(jnp.sum(jnp.exp(logits / n_ensemble), axis=-1))
                
                return (label_logits, log_normalizers, ens_log_normalizers)

            loss_stats = jax.lax.map(loss_stats_subroutine, (x_sb, y_sb))
            return tree_map(lambda z: z.reshape((x.shape[0],)), loss_stats)

        return jax.lax.map(get_subset_loss_stats, state) 

    # -------------------------------------------------------------------------
    model = ResNet18(num_classes=NUM_CLASSES, num_filters=N)

    def create_train_state(params, tx, bs):
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=bs)

    # removed pmap
    init_params = vars_0['params']
    init_bs = vars_0['batch_stats']
    init_opt_state = vmap(vmap(optimizer.init, axis_name='within_subset'), 
                        axis_name='over_subsets')(init_params)
    init_step_state = vmap(vmap(create_train_state, axis_name='within_subset'), 
                        axis_name='over_subsets')(init_params, init_opt_state, init_bs)

    # training loop
    state = init_step_state
    gls = jax.jit(get_loss_stats).lower(init_step_state, jax.ShapeDtypeStruct((tranche_size, 224, 224, 3), jnp.dtype('float32')), jax.ShapeDtypeStruct((tranche_size,), jnp.dtype('int32'))).compile()

    # -------------------------------------------------------------------------
    # create checkpointer
    info('create checkpointer')
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}')

    def save_stats(step, state, train_x, train_y, val_x, val_y):
        train_loss_stats = gls(state, train_x, train_y)
        val_loss_stats = gls(state, val_x, val_y)
        ckpt = {'train': train_loss_stats, 'val': val_loss_stats}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=step, overwrite=False, keep=150,
                                orbax_checkpointer=checkpointer)
        return

    def cleanup_save_stats(step, state, val_x, val_y):
        val_loss_stats = gls(state, val_x, val_y)
        ckpt = {'train': None, 'val': val_loss_stats}
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, 
                                step=step, overwrite=False, keep=150,
                                orbax_checkpointer=checkpointer)
        return
    info('...done')
    # -------------------------------------------------------------------------

    info('Entering training loop...')
    try:
        steps = 0
        prev_record_step = 0
        start = time.time()
        for _ in range(epochs):
            prefetch_iter = prefetch_to_device(train_loader, PREFETCH_N)
            for _, batch in enumerate(prefetch_iter):
                x, y = batch
                state = update(state, x, y)
                steps += num_batches
                if steps > SPACING_MULTIPLIER * prev_record_step:
                    prev_record_step = steps
                    save_stats(steps, state, x, y, X_val, y_val)
                    info(f'step {steps}: elapsed time {time.time() - start}')
    finally:
        cleanup_save_stats(steps, state, X_val, y_val)
        model_ckpt_dir = os.path.join(BASE_SAVE_DIR, f'ens_{n_ensemble}_width_{N}_model')
        checkpoints.save_checkpoint(model_ckpt_dir, state, step=steps, orbax_checkpointer=checkpointer)

    info('...exiting loop.')
    # note that return value is a pytree
    return None, None


def get_n_ensemble(width):
    NUM_ENS_64 = 64
    _exp = np.log2(width) - 6.0
    ens = NUM_ENS_64 // (4 ** _exp)
    return int(ens)


def get_div_size(N: int, ensemble_size: int):
    return min(int((2 ** 11) // N), ensemble_size)


def apply(key, train_loader, val_data, devices, model_params, training_params, N):
    # get sharded keys
    # n_ensemble = model_params['repeat']
    n_ensemble = get_n_ensemble(N)
    div = get_div_size(N, n_ensemble)

    # initialize!
    key, subsample_key = split(key)
    init_keys = split(key, num=n_ensemble)
    del key

    vars_0 = initialize(init_keys, N, div) # shape: (n_ensemble // div, div, param dims)
    info('initialized parameters!')

    # optimizer
    eta_0 = training_params['eta_0']

    adam = optax.adam(eta_0)
    set_to_zero = optax.set_to_zero()
    opt_mapping = {'adam': adam, 'freeze': set_to_zero}
    param_mapping = tree_map(lambda z: 'adam' if z.ndim > 2 else 'freeze', vars_0['params'])
    optimizer = optax.multi_transform(opt_mapping, param_mapping)

    # train!
    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    info('entering train function')
    _ = train(subsample_key, vars_0, N, optimizer, train_loader, *val_data, epochs, batch_size, n_ensemble)

    # result = OnlineResult(key, N, n_ensemble, train_losses, val_losses)
    return None

