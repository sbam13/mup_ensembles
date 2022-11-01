# from functools import partial
# from src.experiment.model.resnet import ResNet18 as _RN18

# CIFAR_ResNet18 = partial(_RN18, n_classes = 1)

from src.experiment.model.resnet import NTK_ResNet18
from src.experiment.model.miniresnet import MiniResNet18
from src.experiment.model.myrtle_net import MyrtleNetwork
from src.experiment.model.vgg import VGG_12