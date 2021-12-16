from enum import Enum, unique
from control.vlogger import VLogger

# dataset type
@unique
class DataSetType(Enum):
    LOWWER = 0
    CIFAR10 = 1
    CelebA = 2 
    ImageNet = 3
    FEMNIST = 4
    UPPER = 5

# Model Type
@unique
class ModelType(Enum):
    LOWWER = 0
    VGG = 1
    RESNET = 2 
    LANET = 3
    UPPER = 4

# Uniform const
CPU = -6
GPU = -66
CPU_STR_LEN = 3
INIT_LR = 0.1
LR_HALF_LIFE = 10000

GLOBAL_LOGGER_PATH = "logs/hrankFL.log"
GLOBAL_LOGGER = VLogger(GLOBAL_LOGGER_PATH).logger

# CIFAR10 const config
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

MAX_ROUND = 10001