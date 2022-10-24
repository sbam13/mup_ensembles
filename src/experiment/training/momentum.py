from typing import Callable, Mapping

import chex
import jax.numpy as jnp
import optax

from flax.core.frozen_dict import FrozenDict
from jax import pmap, tree_map, value_and_grad, vmap
from jax import device_put_replicated, device_put_sharded
from jax.lax import scan

from jax.random import permutation, split
from jaxlib.xla_extension import Device
from src.experiment.model import ResNet18
from src.experiment.training.root_schedule import blocked_polynomial_schedule

# TODO: ensure batching is consistent !!!
# split apply into train and predict

@chex.dataclass
class Result: 
    weight_init_key: chex.PRNGKey
    params_f: chex.ArrayTree
    train_loss_f: chex.ArrayDevice
    test_loss_f: chex.Scalar
    test_deviations_f: chex.ArrayDevice

def initialize(keys: chex.PRNGKey, model, devices: list[Device], data_params: Mapping) -> FrozenDict:
    assert len(keys) == len(devices)

    CIFAR_SHAPE = (32, 32, 3)
    dummy_input = jnp.zeros((1,) + CIFAR_SHAPE) # added batch index
    replicated_dummy = device_put_replicated(dummy_input, devices)

    # get shape data from model
    return pmap(model.init)(keys, replicated_dummy)


def train(loss_fn: Callable, params0: chex.ArrayTree, 
        optimizer: optax.GradientTransformation, Xtr, ytr, keys: chex.PRNGKey, 
        devices: list[Device], epochs: int = 80, batch_size: int = 128) -> tuple[int, 
        chex.ArrayTree, chex.Scalar]:
    num_batches = Xtr.shape[0] // batch_size

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
        opt_state: chex.ArrayTree

    @chex.dataclass
    class DistributedEpochState:
        key: chex.PRNGKey
        model_state: DistributedStepState

    loss_grad_fn = value_and_grad(loss_fn)

    def step(step_state: tuple, data: tuple):
        _, params, opt_state = step_state
        batch, labels = data

        loss_value, grads = loss_grad_fn(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return DistributedStepState(loss_value, params, opt_state), None
    
    @pmap(axis_name='trials')
    def update(state: DistributedEpochState, 
                Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice) -> DistributedEpochState:
        key, other = split(state.key, 2)

        # shuffle
        Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data
        Xtr_sb = Xtr_shuffled.reshape((num_batches, batch_size, *Xtr_shuffled.shape[1:]))
        ytr_sb = ytr_shuffled.reshape((num_batches, batch_size, *ytr_shuffled.shape[1:]))

        # SGD steps over batches in epoch
        model_state_e, _ = scan(step, state.model_state, (Xtr_sb, ytr_sb))

        return DistributedEpochState(other, model_state_e)
    # ----------------------------------------------------------------------
    losses0 = device_put_replicated(jnp.zeros(1), devices)

    init_opt_state = pmap(optimizer.init)(params0)
    init_step_state = DistributedStepState(loss=losses0, params=params0, opt_state=init_opt_state)
    init_epoch_state = DistributedEpochState(key=keys, model_state=init_step_state)

    # training loop
    state = init_epoch_state
    for _ in range(epochs):
        state = update(state, Xtr, ytr)
    
    return state.model_state.params, state.model_state.loss
    

def loss_and_deviation(apply_fn, loss, X_test, y_test):
    """Returns test loss and the deviation (y_hat - y_true)."""
    vapply_fn = vmap(apply_fn)
    vloss = vmap(loss)

    @pmap
    def compute_ld(X_test, y_test):
        yhat = vapply_fn(X_test)
        deviation = yhat - y_test
        test_losses = vloss(y_test, yhat)
        loss = test_losses.mean()
        return loss, deviation
    
    return compute_ld(X_test, y_test)


def apply(key, data, devices, model_params, training_params):
    N = model_params['N']
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)

    # get sharded keys
    shard = lambda key_array: device_put_sharded(tuple(key_array), devices)

    keys = split(key, num=len(devices) * 2)
    del key

    init_keys, apply_keys = shard(keys[:len(devices)]), shard(keys[len(devices),:])

    # get initial parameters
    params_0 = initialize(init_keys, model, devices)

    # compose apply function
    apply_fn = model.apply
    alpha = model_params['alpha']
    shift_apply = lambda params, Xin: alpha * (apply_fn(params, Xin) - apply_fn(params_0, Xin))

    # compose optimizer
    POWER = -0.5
    eta_0 = training_params['eta_0']
    momentum = training_params['momentum']
    batch_size = training_params['batch_size']
    block_steps = 512 // batch_size
    lr_schedule = blocked_polynomial_schedule(eta_0, POWER, block_steps=block_steps)
    
    optimizer = optax.sgd(lr_schedule, momentum)

    # loss function
    mse = lambda y, yhat: jnp.mean((y - yhat) ** 2)
    loss_fn = lambda params, Xin, yin: mse(shift_apply(params, Xin), yin)

    # train!
    epochs = training_params['epochs']
    params_f, train_loss_f = train(loss_fn, params_0, optimizer, 
                                    *data['train'], apply_keys, devices, 
                                    epochs)

    test_loss_f, test_deviations_f = loss_and_deviation(apply_fn, mse, *data['test'])

    parallel_result = Result(weight_init_key=init_keys, params_f=params_f, 
                train_loss_f=train_loss_f, test_loss_f=test_loss_f, 
                test_deviations_f=test_deviations_f)
    
    results = [None] * len(devices)
    for d in range(len(devices)):
        results[d] = tree_map(lambda z: z[d], parallel_result)
    return results
    

