from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

from torch.utils.data import Subset

import torch as ch
import jax.random as jr
import numpy as np

# from src.run.constants import IMAGENET_FOLDER

from logging import getLogger
from typing import Mapping

log = getLogger(__name__)


# class RandomSubsetSampler(Sampler):
#     def __init__(self, data_source: Optional[Sized], P: int, rng: np.random.Generator) -> None:
#         super().__init__(data_source)
#         data_source_len = len(data_source)
#         self.indices = rng.choice(data_source_len, P)

#     def __iter__(self) -> Iterator[T_co]:
#         return iter(self.indices)

#     def __len__(self) -> int:
#         return len(self.indices)


def load_imagenet_data(data_dir: str, data_params: Mapping) -> tuple[ch.utils.data.Dataset, ch.utils.data.Dataset]:
    P = data_params['P']
    val_P = data_params['val_P']
    assert P > 0
    assert val_P > 0

    log.info(f'Loading ImageNet-1k dataset from {data_dir}.')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    channels_last_transform = transforms.Lambda(lambda x: x.permute(1, 2, 0))

    transform_comp = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            channels_last_transform])

    val_transform_comp = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        channels_last_transform])

    train_data = ImageNet(data_dir, 'train', transform=transform_comp)
    val_data = ImageNet(data_dir, 'val', transform=val_transform_comp)

    data_seed = data_params['data_seed']
    
    key = jr.PRNGKey(data_seed)
    k1, k2 = jr.split(key)
    del key

    train_indices = jr.choice(k1, len(train_data), (P,), replace=False)
    train_indices = np.array(train_indices)

    train_data = Subset(train_data, train_indices)

    val_indices = jr.choice(k2, len(val_data), (val_P,), replace=False)
    val_indices = np.array(val_indices)
    
    val_data = Subset(val_data, val_indices)

    log.info(f'Training size: {len(train_data)}\nValidation size: {len(val_data)}')
    return train_data, val_data


