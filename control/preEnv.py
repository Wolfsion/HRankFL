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


@unique
class MessType(Enum):
    LOWWER = 0
    UPLOAD_STATIC_DICT = 1
    DOWNLOAD_STATIC_DICT = -1
    UPLOAD_RANK = 2
    DOWNLOAD_RANK = -2
    UPLOAD_PRUNING_RATE = 3
    DOWNLOAD_PRUNING_RATE = -3
    UPPER = 4


# Uniform const
CPU = -6
GPU = -66
CPU_STR_LEN = 3
INIT_LR = 0.1
LR_HALF_LIFE = 10000
ORIGIN_CP_RATE = [0.] * 100

# simulation
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 20
NUM_CLIENTS = 10

MAX_ROUND = 10001
MAX_DEC_DIFF = 0.3
ADJ_INTERVAL = 50
ADJ_HALF_LIFE = 10000

# CIFAR10 const config
CIFAR10_NAME = "CIFAR10"
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# VGG const config


# Logger
GLOBAL_LOGGER_PATH = "logs/hrankFL.log"
GLOBAL_LOGGER = VLogger(GLOBAL_LOGGER_PATH, True).logger
