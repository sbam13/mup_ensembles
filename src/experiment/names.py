from . import cifar10_resnet
from . import imagenet_resnet

names = {
    ('cifar10', 'resnet18'): cifar10_resnet,
    ('imagenet', 'resnet18'): imagenet_resnet
}