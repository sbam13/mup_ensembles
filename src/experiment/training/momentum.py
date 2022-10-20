from typing import Callable, NamedTuple, Union

import jax.numpy as jnp

from jax.lax import while_loop, scan
from jax.random import PRNGKey, permutation, split
from jax import vmap, value_and_grad

from jaxlib.xla_extension import DeviceArray

import optax

# TODO: ensure batching is consistent !!!
# split apply into train and predict


def initialize(shape):
    raise NotImplementedError


def train(apply_fn: Callable, loss_fn: Callable, params0: optax.Params, 
        optimizer: optax.GradientTransformation, Xtr, ytr, key: PRNGKey, 
        epochs = 100, eta = 0.2, loss_cutoff = 1e-2, batch_size = 128) -> optax.Params:
    
    class StepState(NamedTuple):
        loss: Union[DeviceArray, float]
        params: optax.Params
        opt_state: tuple

    class EpochState(NamedTuple):
        key: PRNGKey
        epoch: int
        model_state: StepState

    def step(step_state: tuple, data: tuple):
        _, params, opt_state = step_state
        batch, labels = data

        loss_value, grads = value_and_grad(loss_fn)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return StepState(loss_value, params, opt_state), None

    def stopping_condition(state: EpochState) -> bool:
        return state.epoch < epochs and state.model_state.loss >= loss_cutoff

    num_batches = Xtr.shape[0] // batch_size
    Xtr_batched = Xtr.reshape((num_batches, batch_size, Xtr.shape[1]))
    ytr_batched = ytr.reshape((num_batches, batch_size, 1))

    def update(state: EpochState) -> EpochState:
        rkey1, rkey2 = split(state.key, 2)
        
        # shuffle
        Xtr_shuffled, ytr_shuffled = permutation(rkey1, Xtr), permutation(rkey1, ytr)
        
        # batch training data
        Xtr_sb = Xtr_shuffled.reshape((num_batches, batch_size, *Xtr_shuffled.shape[1:]))
        ytr_sb = ytr_shuffled.reshape((num_batches, batch_size, *ytr_shuffled.shape[1:]))

        # SGD steps over batches in epoch
        model_state_e, _ = scan(step, state.model_state, (Xtr_sb, ytr_sb))

        return EpochState(rkey2, state.epoch + 1, model_state_e)

    init_opt_state = optimizer.init(params0)
    init_state = EpochState(0, jnp.inf, params0, init_opt_state)

    end_state = while_loop(stopping_condition, update, init_state)
    return end_state.params
    

def predict(apply_fn, mse, X_test, y_test, batch_size = 128):
    num_batches = X_test.shape[0] // batch_size

    batched_X_test = X_test.reshape((num_batches, batch_size, X_test.shape[2]))
    batched_y_test = y_test.reshape((num_batches, batch_size, 1))

    batch_apply_fn = vmap(apply_fn)
    batch_mse = vmap(mse)

    batched_yhat = batch_apply_fn(batched_X_test) # shape: (nb, bs, 1)
    batched_test_loss = batch_mse(batched_y_test, batched_yhat)
    loss = batched_test_loss.mean()
    
    return loss, batched_yhat.reshape((X_test.shape[0], *batched_yhat[2:]))


def apply(key, data, hyperparams):
    training_hyperparams = hyperparams['training_params']
    model_params = hyperparams['model_params']

    key, temp_key = split(key)
    init_params = initialize(temp_key, model_params)
    del temp_key

    train()
    shift_apply = lambda params, Xin: alpha * ( apply_fn(params, Xin) - apply_fn(params0,Xin) )


# def train_models(data, batch_size = 200, num_inits=3, alpha0 = 1.0, N=100, num_iter = 100, eta = 0.2):
#     #print("alpha0 = %0.2f" % alpha)
#     Xtr, ytr = data['train']

#     # all_params0 = []
#     # all_paramsf = []
#     # ytr_2d = jnp.reshape(ytr, (ytr.shape[0],1))
#     # #print(ytr_2d.shape)
#     # all_trains = []
#     # all_test = []

#     alpha = alpha0/jnp.sqrt(N)
#     _, params0 = init_fn(random.PRNGKey(i), (-1,32,32,3)) # new init

#     shift_apply = lambda params, Xin: alpha * ( apply_fn(params, Xin) - apply_fn(params0,Xin) )

#     mse = lambda y, yhat: jnp.mean((y - yhat) ** 2)
#     loss_fn = lambda params, Xin, yin: mse(shift_apply(params, Xin), yin)
#     grad_fn = jit(grad(loss_fn, 0))

#     lr_exp = 2
#     optimizer = optax.sgd(N*eta/alpha0 ** lr_exp, momentum=0.95)
#     opt_state = opt_init(params0)

#     # for 

#     num_batches = Xtr.shape[0]//batch_size

#         print("init %d" % i)
#         for t in range(num_iter):
#             loss_t = 0.0
#             for b in range(num_batches):
#             	opt_state = opt_update(t, grad_fn(get_params(opt_state),Xtr[b*batch_size:(b+1)*batch_size],ytr_2d[b*batch_size:(b+1)*batch_size]), opt_state)
#             	loss_t += 1/num_batches * loss_fn(get_params(opt_state), Xtr[b*batch_size:(b+1)*batch_size],ytr_2d[b*batch_size:(b+1)*batch_size])
#             sys.stdout.write('\r loss: %0.6f' % loss_t)
#             if loss_t < 1e-2:
#                 break
#         all_trains += [loss_t]
#         all_params0 += [params0]
#         all_paramsf += [get_params(opt_state)]

#         partial_apply = partial(shift_apply, get_params(opt_state))

#         X_test, y_test = data['test']

#         loss, yhat = predict(partial_apply, mse, X_test, y_test, batch_size)

#         # tb = 100
#         # num_batch_te = X_te_bin.shape[0] // tb
#         # test_loss = 0.0
#         # for n in range(num_batch_te):
#         #     yhat[i,tb*n:tb*(n+1)] = shift_apply(get_params(opt_state), X_te_bin[tb*n:tb*(n+1)])[:,0]
#         #     test_loss += 1/num_batch_te * loss_fn(get_params(opt_state), X_te_bin[tb*n:(n+1)*tb],y_te_bin[tb*n:(n+1)*tb].reshape((tb,1)))
#         # all_test += [test_loss]
#     test_ens = jnp.mean( (jnp.mean(yhat, axis = 0)-y_te_bin )**2  )

#     # TODO:
#     # np.save(savedir+"ens_test_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), test_ens)
#     # np.save(savedir+"test_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), all_test)
#     # np.save(savedir+"train_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), all_trains)
    
    
#     #np.save(modeldir+"params0_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), all_params0)
#     #np.save(modeldir+"paramsf_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), all_paramsf)
#     #np.save(savedir+"yhat_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), yhat)
#     return

# np.save(savedir +"test_labels", y_te_bin)