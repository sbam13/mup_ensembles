from unittest.mock import patch

import src
from src.experiment.cifar10_resnet import PreprocessDevice
import src.experiment.dataset.cifar10 as cifar10

import jax.numpy as jnp

def test_load_cifar_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

@patch('src.run.PreprocessDevice.PreprocessDevice.__init__')
def test_normalize_data():
    Xtrain = jnp.ones((24, 32, 32, 3))
    Xtest = jnp.ones((12, 32, 32, 3))
    ytrain = jnp.repeat(jnp.arange(0, 4), 6)
    ytest = jnp.repeat(jnp.arange(5, 9), 3)
    d = {'train': (Xtrain, ytrain), 'test': (Xtest, ytest)}


def test_pd_file_access(tmp_path):
    src.run.Prepr
    data_params =  {'root_dir': 'dd', 'other': 'other_val'}
    pd = PreprocessDevice('sd', data_params, True)

    assert pd.save_dir == 'sd'
    assert pd.data_dir == 'dd'
    assert 