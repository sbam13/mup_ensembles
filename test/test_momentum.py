from src.experiment.model import ResNet18
from jax.random import PRNGKey
import jax
import jax.numpy as jnp
from jax.random import split
from src.experiment.model.common import NTK_Conv

from src.experiment.training.momentum import apply, initialize, loss_and_deviation
from src.experiment.training.sd_alpha_test import apply as sd_apply

import flax.linen as nn

def test_model():
    N = 4
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)
    key = PRNGKey(12)
    # kk = jax.device_put_replicated(key, jax.devices())
    dummy = jnp.ones((1, 32, 32, 3))
    params = model.init(key, dummy)
    y = model.apply(params, dummy)
    print(y, y.shape, type(y))


def test_pmap_model():
    N = 200
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)
    key = PRNGKey(22622)
    key2 = PRNGKey(233923)
    keys = jax.random.split(key, 2)
    inner_keys = jax.random.split(key2, 2)
    tot = 0
    for key in keys:
        for key2 in keys:
            kk = jax.device_put_replicated(key, jax.devices())
            # dummy = jnp.ones((1, 32, 32, 3))
            devices = jax.devices()
            params = initialize(kk, model, devices)
            # unshard = jax.tree_map(lambda z: z[0], params)
            papply = jax.pmap(model.apply)
            rand_norm = jax.random.normal(key2, (10000, 32, 32, 3))
            f = lambda z: z / jnp.sqrt(jnp.sum(z ** 2))
            spherical = jax.vmap(f)(rand_norm)

            # dumdum = jnp.ones((1, 32, 32, 3)) / 3072.0
            dd = jax.device_put_replicated(spherical, devices)
            tot += jnp.mean(papply(params, dd)) ** 2
    return tot / 4

    # Xt = jnp.ones((5, 32, 32, 3)) / 3072.0
    # print(unshard)
    # print(jnp.abs(model.apply(unshard, Xt)))
    # yt = jnp.ones((5, 1))

    # XtXt = jax.device_put_replicated(Xt, devices)
    # ytyt = jax.device_put_replicated(yt, devices)

    # mse = lambda y, yhat: jnp.mean((y - yhat) ** 2)
    # loss, deviations = loss_and_deviation(model.apply, params, mse, XtXt, ytyt)
    # return loss, deviations

def test_ld():
    devices = jax.devices()

    key = PRNGKey(22622)
    kk = jax.device_put_replicated(key, devices)

    Xt = jnp.ones((5, 32, 32, 3)) / 3072.0
    yt = jnp.ones((5, 1))

    XtXt = jax.device_put_replicated(Xt, devices)
    ytyt = jax.device_put_replicated(yt, devices)

    params = jax.device_put_replicated(jnp.ones(1), devices)
    p_invariant = jax.vmap(lambda x: jnp.array([0.5]))
    f = lambda p, x: p_invariant(x)
    mse = lambda y, yhat: jnp.mean((y - yhat) ** 2)
    loss, deviations = loss_and_deviation(f, mse, jnp.ones(1), XtXt, ytyt)
    return loss, deviations


def test_ntk_conv():
    l = NTK_Conv(3, (2, 2), kernel_init=nn.initializers.normal(1.0))
    l2 =  NTK_Conv(3, (2, 2))
    l3 = nn.Conv(3, (2, 2))
    key = PRNGKey(24)
    inp = jnp.ones((1, 6, 6, 234))
    p = l.init(key, inp)
    # p2 = l2.init(key, inp)
    print(p)
    kp = p['params']['kernel']
    print(jnp.sum(kp ** 2) / kp.size)
    # print(l.apply(p, inp), l3.apply(p, inp))


def test_apply():
    key = jax.random.PRNGKey(12)
    X = jax.random.normal(key, (8, 32, 32, 3)) 
    y = jnp.arange(8.0).reshape((8, 1))
    
    X -= jnp.mean(X, axis=0)
    g = lambda W: W / jnp.sum(W ** 2, dtype=jnp.float32)
    v_g = jax.vmap(g)
    X = v_g(X)
    data = {'train': (X, y), 'test': (X.copy(), y.copy())}
    devices = jax.devices()
    sharded_data = jax.device_put_replicated(data, devices)
    mp = {'N': 64, 'alpha': 0.5}
    tp = {'eta_0': 1e-2, 'momentum': 0.9, 'batch_size': 4, 'epochs': 2}
    return apply(key, sharded_data, devices, mp, tp)


def test_apply_single():
    key = jax.random.PRNGKey(14432)
    X = jax.random.normal(key, (256, 32, 32, 3)) 
    y = jax.random.rademacher(key, (256, 1),dtype=jnp.float32)
    
    X /= jnp.linalg.norm(X, axis=0)
    data = {'train': (X, y), 'test': (X.copy(), y.copy())}
    mp = {'N': 64, 'alpha': 0.01}
    tp = {'eta_0': 1e-2, 'momentum': 0.9, 'batch_size': 128, 'epochs': 2}
    return sd_apply(key, data, mp, tp)




def test_dims():
    key = jax.random.PRNGKey(12)
    X, y = jnp.ones((8, 32, 32, 3)), jnp.ones((8, 1))
    data = {'train': (X, y), 'test': (X.copy(), y.copy())}
    devices = jax.devices()
    sharded_data = jax.device_put_replicated(data, devices)
    mp = {'N': 64, 'alpha': 0.5}
    tp = {'eta_0': 1.0, 'momentum': 0.9, 'batch_size': 4, 'epochs': 2}

    N = mp['N']
    hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
    model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)

    # get sharded keys
    keys = split(key, num=len(devices) * 2)
    del key

    shard = lambda key_array: jax.device_put_sharded(tuple(key_array), devices)
    init_keys, apply_keys = shard(keys[:len(devices)]), shard(keys[len(devices):])

    # get initial parameters
    params_0 = initialize(init_keys, model, devices)
    unshard_params = jax.tree_map(lambda z:z[0], params_0)
    yhat = model.apply(unshard_params, X)
    return yhat

if __name__ == '__main__':
    results = test_apply_single()
    print(results)
    # print(test_dims())

