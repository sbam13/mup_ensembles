from src.experiment.model import ResNet18
from jax.random import PRNGKey
import jax

from src.experiment.training.momentum import initialize

N = 4
hidden_sizes = (N, 2 * N, 4 * N, 8 * N)
model = ResNet18(hidden_sizes=hidden_sizes, n_classes=1)

key = PRNGKey(12)
kk = jax.device_put_replicated(key, jax.devices())
sharded_params = initialize(kk, model, jax.devices())