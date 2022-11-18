import chex

@chex.dataclass
class Result: 
    weight_init_key: chex.PRNGKey
    params_f: chex.ArrayTree
    train_losses: chex.ArrayTree
    test_losses: chex.ArrayTree
    test_loss_f: chex.Scalar
    test_yhat_f: chex.ArrayTree
    test_y: chex.ArrayTree
    num_epochs: chex.ArrayTree