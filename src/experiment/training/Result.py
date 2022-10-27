import jax.numpy as jnp
import jax
import chex

@chex.dataclass
class Result: 
    weight_init_key: chex.PRNGKey
    params_f: chex.ArrayTree
    train_losses: chex.ArrayTree
    test_loss_f: chex.Scalar
    test_deviations_f: chex.ArrayDevice