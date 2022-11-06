from functools import partial
from typing import Callable
from logging import info

import chex
import jax.numpy as jnp
import optax

from flax.core.frozen_dict import FrozenDict
from jax import pmap, tree_map, value_and_grad
from jax import device_put_replicated, device_put_sharded, device_get
from jax.lax import scan

from jax.random import permutation, split
from jaxlib.xla_extension import Device
from src.experiment.training.Result import Result
from src.experiment.training.root_schedule import blocked_polynomial_schedule

# from src.experiment.model.resnet import NTK_ResNet18
from src.experiment.model.wide_resnet import WideResnet

# MSE loss function
def mse(y, yhat):
    return jnp.mean((y - yhat) ** 2)


def initialize(keys: chex.PRNGKey, init_fn, devices: list[Device]) -> chex.ArrayTree:
    assert len(keys) == len(devices)

    CIFAR_SHAPE = (32, 32, 3)
    # dummy_input = jnp.zeros((1,) + CIFAR_SHAPE) # added batch index
    # replicated_dummy = device_put_replicated(dummy_input, devices)
    
    def get_params(key):
        return init_fn(key, (1,) + CIFAR_SHAPE)

    return pmap(get_params)(keys)


def train(apply_fn: Callable, params0: chex.ArrayTree, 
        optimizer: optax.GradientTransformation, Xtr, ytr, X_test, y_test, keys: chex.PRNGKey, 
        alpha: chex.Scalar, epochs: int = 80, batch_size: int = 128) -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    num_batches = Xtr.shape[1] // batch_size # 0 is sharding dimension

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
        params: chex.ArrayTree
        opt_state: chex.ArrayTree

    @chex.dataclass
    class DistributedEpochState:
        key: chex.PRNGKey
        model_state: DistributedStepState
    
    shape, values = params0
    centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn((shape, values), Xin))


    @partial(pmap)
    def compute_loss(state: DistributedEpochState, Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice):
        """Computes the loss of the model at state `state` with data `(Xtr, ytr)`."""
        params = state.model_state.params
        return mse(centered_apply((shape, params), Xtr), ytr)

    @partial(pmap)
    def update(state: DistributedEpochState, 
                Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice) -> DistributedEpochState:
        """Runs one epoch."""
        key, other = split(state.key, 2)
        
        # loss and gradient functions -------------------------------
        alpha_scaled_loss = lambda y, yhat: (1.0 / alpha ** 2) * mse(y, yhat)
 
        def loss_fn(params, Xin, yin):
            combined = (shape, params)
            return alpha_scaled_loss(centered_apply(combined, Xin), yin)
        
        loss_grad_fn = value_and_grad(loss_fn, argnums=0)
        # -----------------------------------------------------------

        def step(step_state: DistributedStepState, data: tuple) -> DistributedStepState:
            """Takes an SGD step."""
        # unpack
            params = step_state.params
            opt_state = step_state.opt_state
            batch, labels = data

            # update params
            # param_shape = tree_map(lambda z: z.shape, params)
            # data_shape = tree_map(lambda z: z.shape, batch)
            _, grads = loss_grad_fn(params, batch, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            
            # apply updates only to mutable params
            updated_params = optax.apply_updates(params, updates)

            return DistributedStepState(params=updated_params, opt_state=opt_state), None
    

        # shuffle
        Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data
        Xtr_sb = Xtr_shuffled.reshape((num_batches, batch_size, *Xtr_shuffled.shape[1:]))
        ytr_sb = ytr_shuffled.reshape((num_batches, batch_size, *ytr_shuffled.shape[1:]))

        # SGD steps over batches in epoch
        model_state_e, _ = scan(step, state.model_state, (Xtr_sb, ytr_sb))

        return DistributedEpochState(key=other, model_state=model_state_e)
    
    # ----------------------------------------------------------------------

    init_opt_state = pmap(optimizer.init)(values)
    init_step_state = DistributedStepState(params=values, opt_state=init_opt_state) # TODO: question? is using params0 in both screwing things up?
    init_epoch_state = DistributedEpochState(key=keys, model_state=init_step_state)

    # training loop
    state = init_epoch_state
    losses = []
    test_losses = []
    info('Entering training loop...')
    for e in range(epochs):
        state = update(state, Xtr, ytr)
        if e % 5 == 0:
            losses.append(compute_loss(state, Xtr, ytr))
            test_losses.append(compute_loss(state, X_test, y_test))
    info('...exiting loop.')
    # note that return value is a pytree
    return (shape, state.model_state.params), losses, test_losses
    

def loss_and_yhat(apply_fn, alpha, params, params_0, X_test, y_test):
    """Returns test loss and the predictions yhat."""
    # vapply_fn = vmap(apply_fn)
    # vloss = vmap(loss)

    def compute_ld(params, p0, X_test, y_test):
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        yhat = centered_apply(params, X_test)
        test_loss = mse(y_test, yhat)
        return test_loss, yhat
    
    return pmap(compute_ld)(params, params_0, X_test, y_test)


def apply(key, data, devices, model_params, training_params):
    N = model_params['N']
    
    # MODELS --------------------------------------------------------
    # hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    # hidden_sizes = (N, 2 * N)
    init_fn, apply_fn, _ =  WideResnet(block_size=4, k=(N // 16), num_classes=1)
    # ---------------------------------------------------------------


    # get sharded keys
    keys = split(key, num=len(devices) * 2)
    del key

    shard = lambda key_array: device_put_sharded(tuple(key_array), devices)
    init_keys, apply_keys = shard(keys[:len(devices)]), shard(keys[len(devices):])

    # get initial parameters
    params_0 = initialize(init_keys, init_fn, devices)
    # print(tree_map(lambda z: z.shape, params_0))

    # compose optimizer
    # def zero_grads():
    #     def init_fn(_): 
    #         return ()
    #     def update_fn(updates, state, params=None):
    #         return tree_map(jnp.zeros_like, updates), ()
    #     return optax.GradientTransformation(init_fn, update_fn)

    batch_size = training_params['batch_size']
    eta_0 = training_params['eta_0']
    # weight_decay = training_params['weight_decay'] * batch_size
    momentum = training_params['momentum']
    
    # POWER = -0.5
    # LR_DROP_STAGE_SIZE = 512
    
    # block_steps = LR_DROP_STAGE_SIZE // batch_size
    # lr_schedule = blocked_polynomial_schedule(eta_0, POWER, block_steps=block_steps)
    # optimizer = optax.sgd(lr_schedule, momentum)
    # adam = optax.adam(eta_0)
    sgd_fixed_eta = optax.sgd(eta_0, momentum)
    optimizer = sgd_fixed_eta
    # optimizer = optax.multi_transform({'sgd': sgd_fixed_eta, 'zero': zero_grads()},
    #                                     {'params': 'sgd', 'scaler': 'zero'})
    # optimizer = optax.adamw(eta_0, weight_decay=weight_decay)

    # compose apply function
    alpha = model_params['alpha']

    # train!
    epochs = training_params['epochs']
    P = data['train'][0].shape[1] # 0 is sharding dimension
    if P % batch_size != 0:
        raise ValueError(f'Batch size of {batch_size} does not divide training data size {P}.')

    params_f, train_losses, test_losses = train(apply_fn, params_0, optimizer, 
                                    *data['train'], *data['test'], apply_keys,
                                    alpha, epochs, batch_size)

    test_loss_f, test_yhat_f = loss_and_yhat(apply_fn, 
                                            alpha, params_f, params_0, 
                                            *data['test'])

    parallel_result = Result(weight_init_key=init_keys, params_f=params_f, 
                train_losses=train_losses, test_losses=test_losses, test_loss_f=test_loss_f, 
                test_yhat_f=test_yhat_f, test_y=data['test'][1])
    
    results = [None] * len(devices)
    for d in range(len(devices)):
        results[d] = tree_map(lambda z: z[d], parallel_result)
    return results
    

