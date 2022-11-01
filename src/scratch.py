import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond

def whiten_data(data: dict):
    """Center and normalize so that x-values are on the sphere."""
    X0, y = data['train']
    X_test0, y_test = data['test']

    # center using training mean
    train_mean = jnp.mean(X0, axis=0)
    X0 -= train_mean
    X_test0 -= train_mean

    # normalize
    def normalize(W): 
        # TODO: make this jax.lax.cond
        im_norm = jnp.sum(W ** 2, dtype=jnp.float32)
        div_by_scalar = lambda z, c: z / c 
        id = lambda z, c: z
        return cond(jnp.isclose(im_norm, 0.0), id, div_by_scalar, W, jnp.sqrt(im_norm))

    v_normalize = jit(vmap(normalize))

    X = v_normalize(X0)
    X_test = v_normalize(X_test0)

    # Classes [0 - 4] are 1, classes [5 - 9] are -1
    cifar2 = lambda labels: 2. * ((labels < 5).astype(jnp.float32)) - 1.

    y, y_test = map(cifar2, (y, y_test))

    return dict(train=(X, y), test=(X_test, y_test))