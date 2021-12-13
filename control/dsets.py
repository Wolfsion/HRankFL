from enum import Enum, unique

# dataset type
@unique
class DataSetType(Enum):
    LOWWER = 0
    CIFAR10 = 1
    CelebA = 2 
    ImageNet = 3
    FEMNIST = 4
    UPPER = 5

# CIFAR10 const config
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

MAX_ROUND = 10001