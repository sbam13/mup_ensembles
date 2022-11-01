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
from src.experiment.model.resnet import NTK_ResNet18
from src.experiment.training.Result import Result
from src.experiment.training.root_schedule import blocked_polynomial_schedule


# TODO: ensure batching is consistent !!!
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
        params: chex.ArrayTree
        opt_state: chex.ArrayTree

    @chex.dataclass
    class DistributedEpochState:
        train_loss: chex.Scalar
        key: chex.PRNGKey
        p0: chex.ArrayTree # params at time 0
        model_state: DistributedStepState

    def update(state: DistributedEpochState, 
                Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice) -> DistributedEpochState:
        """Runs one epoch."""
        key, other = split(state.key, 2)
        p0 = state.p0
        # p0_shape = tree_map(lambda z:z.shape, p0)
        
        # loss and gradient functions -------------------------------
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        
        alpha_scaled_loss = lambda y, yhat: (1.0 / alpha ** 2) * mse(y, yhat)
        
        def loss_fn(mut_p, immut_p, Xin, yin):
            combined = {'params': mut_p, 'scaler': immut_p}
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
            mutable_p, immutable_p = params['params'], params['scaler']
            _, grads = loss_grad_fn(mutable_p, immutable_p, batch, labels)
            grad_norms = tree_map(lambda z: jnp.linalg.norm(z), grads)
            updates, opt_state = optimizer.update(grads, opt_state, mutable_p)
            
            # apply updates only to mutable params
            mutable_p = optax.apply_updates(mutable_p, updates)
            updated_params = params.copy({'params': mutable_p})

            batch_yhat = centered_apply(updated_params, batch)

            return DistributedStepState(params=updated_params, opt_state=opt_state), None
    

        # shuffle
        Xtr_shuffled, ytr_shuffled = permutation(key, Xtr), permutation(key, ytr)
        
        # batch training data
        Xtr_sb = Xtr_shuffled.reshape((num_batches, batch_size, *Xtr_shuffled.shape[1:]))
        ytr_sb = ytr_shuffled.reshape((num_batches, batch_size, *ytr_shuffled.shape[1:]))

        # SGD steps over batches in epoch
        step_state = state.model_state
        for b in range(num_batches):
            step_state, _ = step(step_state, (Xtr_sb[b], ytr_sb[b]))

        # compute train loss
        epoch_params = step_state.params
        epoch_yhat = centered_apply(epoch_params, Xtr)
        epoch_train_loss = alpha_scaled_loss(epoch_yhat, ytr)

        return DistributedEpochState(train_loss=epoch_train_loss, key=other, 
                                    p0=state.p0, model_state=step_state)
    
    # ----------------------------------------------------------------------
    losses0 = jnp.array(0.0)

    init_opt_state = optimizer.init(params0['params'])
    init_step_state = DistributedStepState(params=params0, opt_state=init_opt_state) # TODO: question? is using params0 in both screwing things up?
    init_epoch_state = DistributedEpochState(train_loss=losses0, key=keys, p0=params0, model_state=init_step_state)

    # training loop
    state = init_epoch_state
    losses = [None] * epochs
    info('Entering training loop...')
    for e in range(epochs):
        state = update(state, Xtr, ytr)
        losses[e] = state.train_loss
    info('...exiting loop.')
    # note that return value is a pytree
    return state.model_state.params, losses
    

def loss_and_deviation(apply_fn, alpha, params, params_0, X_test, y_test):
    """Returns test loss, y_hat, and the deviation (y_hat - y_true)."""
    # vapply_fn = vmap(apply_fn)
    # vloss = vmap(loss)

    def compute_ld(params, p0, X_test, y_test):
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        yhat = centered_apply(params, X_test)
        deviation = yhat - y_test
        test_loss = mse(y_test, yhat)
        return test_loss, yhat, deviation
    
    return compute_ld(params, params_0, X_test, y_test)


def apply(key, data, model_params, training_params):
    N = model_params['N']
    hidden_sizes = (N, 2 * N)
    model = NTK_ResNet18(use_bias=False, hidden_sizes=hidden_sizes, stage_sizes=(2, 2), n_classes=1)

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
    # weight_decay = training_params['weight_decay'] * batch_size
    # momentum = training_params['momentum']
    # lr_schedule = blocked_polynomial_schedule(eta_0, POWER, block_steps=block_steps)
    # optimizer = optax.sgd(lr_schedule, momentum)
    optimizer = optax.adam(eta_0)
    # optimizer = optax.adamw(eta_0, weight_decay=weight_decay)

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

    test_loss_f, test_yhat, test_deviations_f = loss_and_deviation(apply_fn, 
                                            alpha, params_f, params_0, 
                                            *data['test'])

    result = Result(weight_init_key=init_keys, params_f=params_f, 
                train_losses=train_losses, test_loss_f=test_loss_f, 
                test_deviations_f=test_deviations_f)
    
    return result
    

