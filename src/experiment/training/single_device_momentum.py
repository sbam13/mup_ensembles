from functools import partial
from typing import Callable
from wsgiref.util import setup_testing_defaults

import chex
import jax.numpy as jnp
import optax

from flax.core.frozen_dict import FrozenDict, freeze
from jax import pmap, tree_map, value_and_grad, vmap
from jax import device_put_replicated, device_put_sharded
from jax.lax import scan

from jax.random import permutation, split
from jaxlib.xla_extension import Device
from src.experiment.model import ResNet18
from src.experiment.training.root_schedule import blocked_polynomial_schedule

# TODO: 
# ensure batching is consistent !!!
# split apply into train and predict

# MSE loss function
mse = lambda y, yhat: jnp.mean((y - yhat) ** 2)


def initialize(keys: chex.PRNGKey, model) -> FrozenDict:
    CIFAR_SHAPE = (32, 32, 3)
    dummy_input = jnp.zeros((1,) + CIFAR_SHAPE) # added batch index

    # get shape data from model
    return model.init(keys, dummy_input)


def train(apply_fn: Callable, params0: chex.ArrayTree, 
        optimizer: optax.GradientTransformation, Xtr, ytr, keys: chex.PRNGKey, 
        alpha: chex.Scalar, epochs: int = 80, batch_size: int = 128) -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    num_batches = Xtr.shape[0] // batch_size # 0 is sharding dimension

    # ----------------------------------------------------------------------
    # @chex.dataclass
    # class StepState:
    #     loss: chex.Scalar
    #     params: chex.ArrayTree
    #     opt_state: optax.OptState

    # @chex.dataclass
    # class EpochState:
    #     key: PRNGKey
    #     model_state: StepState
    
    # distributed pytrees
    @chex.dataclass
    class DistributedStepState:
        loss: chex.Array
        params: chex.ArrayTree
        p0: chex.ArrayTree # params at time 0
        opt_state: chex.ArrayTree

    @chex.dataclass
    class DistributedEpochState:
        key: chex.PRNGKey
        model_state: DistributedStepState

    def update(state: DistributedEpochState, 
                Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice) -> DistributedEpochState:
        key, other = split(state.key, 2)
        p0 = state.model_state.p0
        # p0_shape = tree_map(lambda z:z.shape, p0)
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        def loss_fn(mut_p, immut_p, Xin, yin):
            combined = {'params': mut_p, 'scaler': immut_p}
            return mse(centered_apply(combined, Xin), yin)
        loss_grad_fn = value_and_grad(loss_fn, argnums=0)

        def step(step_state: DistributedStepState, data: tuple):
        # unpack
            params = step_state.params
            opt_state = step_state.opt_state
            batch, labels = data

            # update params
            # param_shape = tree_map(lambda z: z.shape, params)
            # data_shape = tree_map(lambda z: z.shape, batch)
            mutable_p, immutable_p = params['params'], params['scaler']
            batch_yhat = centered_apply(params, batch)
            loss_value, grads = loss_grad_fn(mutable_p, immutable_p, batch, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            mutable_p = optax.apply_updates(mutable_p, updates)
            updated_params = params.copy(dict(params=mutable_p))
            chex.assert_tree_all_finite(updated_params)
            # apply updates only to 
            return DistributedStepState(loss=loss_value, params=updated_params, p0=step_state.p0, opt_state=opt_state), None
    

        # shuffle
        Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data
        Xtr_sb = Xtr_shuffled.reshape((num_batches, batch_size, *Xtr_shuffled.shape[1:]))
        ytr_sb = ytr_shuffled.reshape((num_batches, batch_size, *ytr_shuffled.shape[1:]))

        # SGD steps over batches in epoch
        step_state = state.model_state
        for b in range(num_batches):
            step_state, _ = step(step_state, (Xtr_sb[b], ytr_sb[b]))

        return DistributedEpochState(key=other, model_state=step_state)
    
    # ----------------------------------------------------------------------
    losses0 = jnp.array(0.0)

    init_opt_state = optimizer.init(params0['params'])
    init_step_state = DistributedStepState(loss=losses0, params=params0, p0=params0, opt_state=init_opt_state)
    init_epoch_state = DistributedEpochState(key=keys, model_state=init_step_state)

    # training loop
    state = init_epoch_state
    losses = [None] * epochs
    for e in range(epochs):
        state = update(state, Xtr, ytr)
        losses[e] = state.model_state.loss
    
    # note that return value is a pytree
    return state.model_state.params, losses
    

def loss_and_deviation(apply_fn, alpha, params, params_0, X_test, y_test):
    """Returns test loss and the deviation (y_hat - y_true)."""
    # vapply_fn = vmap(apply_fn)
    # vloss = vmap(loss)

    def compute_ld(params, p0, X_test, y_test):
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        yhat = centered_apply(params, X_test)
        deviation = yhat - y_test
        test_loss = mse(y_test, yhat)
        return test_loss, deviation
    
    return compute_ld(params, params_0, X_test, y_test)


def apply(key, data, model_params, training_params):
    N = model_params['N']
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)

    # get sharded keys
    keys = split(key, num=2)
    del key

    init_keys, apply_keys = keys[0], keys[1]

    # get initial parameters
    params_0 = initialize(init_keys, model)
    # print(tree_map(lambda z: z.shape, params_0))

    # compose optimizer
    POWER = -0.5
    LR_DROP_STAGE_SIZE = 512
    batch_size = training_params['batch_size']
    block_steps = LR_DROP_STAGE_SIZE // batch_size
    eta_0 = training_params['eta_0']
    momentum = training_params['momentum']
    lr_schedule = blocked_polynomial_schedule(eta_0, POWER, block_steps=block_steps)
    optimizer = optax.sgd(lr_schedule, momentum)

    # compose apply function
    apply_fn = model.apply
    alpha = model_params['alpha']

    # train!
    epochs = training_params['epochs']
    P = data['train'][0].shape[0] # 0 is sharding dimension
    if P % batch_size != 0:
        raise ValueError(f'Batch size of {batch_size} does not divide training data size {P}.')
    params_f, train_losses = train(apply_fn, params_0, optimizer, 
                                    *data['train'], apply_keys,
                                    alpha, epochs, batch_size)

    test_loss_f, test_deviations_f = loss_and_deviation(apply_fn, 
                                            alpha, params_f, params_0, 
                                            *data['test'])

    result = Result(weight_init_key=init_keys, params_f=params_f, 
                train_losses=train_losses, test_loss_f=test_loss_f, 
                test_deviations_f=test_deviations_f)
    
    return result
    

