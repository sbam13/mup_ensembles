from torchvision.datasets import CIFAR10

import sys

def download_cifar10(loc):
    training_data = CIFAR10(loc, train=True, download=True)
    test_data = CIFAR10(loc, train=False, download=True)


if __name__ == '__main__':
    loc = sys.argv[1]
    download_cifar10(loc)