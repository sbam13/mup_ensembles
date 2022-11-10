from functools import partial
from typing import Callable
from logging import info

import chex
import jax.numpy as jnp
import math
import optax

import flax.linen as nn

from jax import jit, vmap, pmap, tree_map, value_and_grad
from jax import device_put_replicated, device_put_sharded, device_get
from jax.lax import scan, map as lax_map

from jax.random import permutation, split
from jaxlib.xla_extension import Device
from src.experiment.training.Result import Result
from src.experiment.training.root_schedule import blocked_polynomial_schedule

# from src.experiment.model.resnet import NTK_ResNet18
# from src.experiment.model.vgg import VGG_12
from src.experiment.model.miniresnet import WideResnet

# MSE loss function
def mse(y, yhat):
    return jnp.mean((y - yhat) ** 2)


def initialize(keys: chex.PRNGKey, model, devices: list[Device]) -> chex.ArrayTree:
    assert len(keys) == len(devices)

    CIFAR_SHAPE = (32, 32, 3)
    # GRAY_CIFAR_SHAPE = (32, 32, 1) # change if going back to rgb
    dummy_input = jnp.zeros((1,) + CIFAR_SHAPE) # added batch index
    replicated_dummy = device_put_replicated(dummy_input, devices)
    
    def get_params(key, dummy):
        w_frozen = model.init(key, dummy)
        return w_frozen.unfreeze()

    return pmap(get_params)(keys, replicated_dummy)


def train(apply_fn: Callable, params0: chex.ArrayTree, 
        optimizer: optax.GradientTransformation, Xtr, ytr, X_test, y_test, keys: chex.PRNGKey, 
        alpha: chex.Scalar, epochs: int = 80, batch_size: int = 128) -> tuple[chex.ArrayTree, list[chex.ArraySharded]]:
    P = Xtr.shape[1]
    num_batches = P // batch_size # 0 is sharding dimension

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
        p0: chex.ArrayTree # params at time 0
        model_state: DistributedStepState
    

    def compute_loss(state: DistributedEpochState, Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice, batch_size):
        """Computes the loss of the model at state `state` with data `(Xtr, ytr)`."""
        p0 = state.p0
        params = state.model_state.params
        
        X_batched = Xtr.reshape((-1, batch_size, *Xtr.shape[1:]))
        y_batched = ytr.reshape(((-1, batch_size, *ytr.shape[1:])))
        
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        
        compute_batch_loss = lambda z: mse(centered_apply(params, z[0]), z[1])
        
        return jnp.mean(lax_map(compute_batch_loss, (X_batched, y_batched)))

    MAX_LOSS_COMPUTE_BATCH_SIZE = 8192
    MAX_TEST_LOSS_COMPUTE_BATCH_SIZE = 8000
    loss_compute_batch_size = min(MAX_LOSS_COMPUTE_BATCH_SIZE, P)
    compute_train_loss = pmap(partial(compute_loss, batch_size=loss_compute_batch_size))
    compute_test_loss = pmap(partial(compute_loss, batch_size=MAX_TEST_LOSS_COMPUTE_BATCH_SIZE))

    @partial(pmap)
    def update(state: DistributedEpochState, 
                Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice) -> DistributedEpochState:
        """Runs one epoch."""
        key, other = split(state.key, 2)
        p0 = state.p0
        # p0_shape = tree_map(lambda z:z.shape, p0)
        
        # loss and gradient functions -------------------------------
        centered_apply = lambda vars, Xin: alpha * (apply_fn(vars, Xin) - apply_fn(p0, Xin))
        alpha_scaled_loss = lambda y, yhat: (1.0 / alpha ** 2) * mse(y, yhat)
 
        def loss_fn(combined, Xin, yin):
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

        return DistributedEpochState(key=other, p0=state.p0, model_state=model_state_e)
    
    # ----------------------------------------------------------------------

    init_opt_state = pmap(optimizer.init)(params0)
    init_step_state = DistributedStepState(params=params0, opt_state=init_opt_state) # TODO: question? is using params0 in both screwing things up?
    init_epoch_state = DistributedEpochState(key=keys, p0=params0, model_state=init_step_state)

    # training loop
    state = init_epoch_state
    losses = []
    test_losses = []
    info('Entering training loop...')
    for e in range(epochs):
        state = update(state, Xtr, ytr)
        if e % 5 == 0:
            losses.append(compute_train_loss(state, Xtr, ytr))
            test_losses.append(compute_test_loss(state, X_test, y_test))
    info('...exiting loop.')
    # note that return value is a pytree
    return state.model_state.params, losses, test_losses
    

def loss_and_yhat(apply_fn, alpha, params, params_0, X_test, y_test):
    """Returns test loss and the predictions yhat."""
    # vapply_fn = vmap(apply_fn)
    # vloss = vmap(loss)
    BATCH_SIZE = 500

    def compute_ld(params, p0, X, y):
        X_batched = X.reshape((-1, BATCH_SIZE, *X.shape[1:]))
        y_batched = y.reshape(((-1, BATCH_SIZE, *y.shape[1:])))
        
        centered_apply = lambda Xin: alpha * (apply_fn(params, Xin) - apply_fn(p0, Xin))
        
        def ly(a, b):
            bhat = centered_apply(a)
            return mse(b, bhat), bhat

        vmap_ly = vmap(ly)
        
        batch_losses, batch_yhats = vmap_ly(X_batched, y_batched)
        return jnp.mean(batch_losses), batch_yhats.reshape((-1, *y.shape[1:]))
    
    return pmap(compute_ld)(params, params_0, X_test, y_test)


def apply(key, data, devices, model_params, training_params):
    N = model_params['N']
    
    # MODELS --------------------------------------------------------
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    # hidden_sizes = (N, 2 * N)
    # model = NTK_ResNet18(hidden_sizes=hidden_sizes, stage_sizes=(2, 2), n_classes=1)
    
    # model = MiniResNet18(num_classes=1, num_filters=N)
    
    # model = VGG_12(N)
    model = WideResnet(1, N // 16, 1, conv_init=nn.initializers.normal(math.sqrt(2.0)))

    # model = MyrtleNetwork(N, depth=5)

    # model = ResNet18(n_classes=1)
    # ---------------------------------------------------------------


    # get sharded keys
    keys = split(key, num=len(devices) * 2)
    del key

    shard = lambda key_array: device_put_sharded(tuple(key_array), devices)
    init_keys, apply_keys = shard(keys[:len(devices)]), shard(keys[len(devices):])

    # get initial parameters
    params_0 = initialize(init_keys, model, devices)
    # print(tree_map(lambda z: z.shape, params_0))

    # compose optimizer
    def zero_grads():
        def init_fn(_): 
            return ()
        def update_fn(updates, state, params=None):
            return tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)

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
    optimizer = optax.multi_transform({'sgd': sgd_fixed_eta, 'zero': zero_grads()},
                                        {'params': 'sgd', 'scaler': 'zero'})
    # optimizer = optax.adamw(eta_0, weight_decay=weight_decay)

    # compose apply function
    apply_fn = model.apply
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
    

