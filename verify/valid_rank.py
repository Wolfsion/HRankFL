import pickle
from copy import deepcopy
from os.path import join

import torch
from torch.nn.functional import one_hot
from fedlab.utils.dataset import CIFAR10Partitioner
from torch.utils.data import Sampler, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from collections import OrderedDict

from model import modelUtil
from pruning.vhrank import VGG16HRank
from env.runtimeEnv import *

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
CLASSES = 10
ORIGIN_CP_RATE = [0.] * 100
VGG16 = join(vgg_model, "vgg_16_bn.pt")

num_clients = 2
batch_size = 32

dataset = None
indices_dict = OrderedDict()
samplers_dict = OrderedDict()
loaders_dict = OrderedDict()
hranks_dict = OrderedDict()
ranks_dict = OrderedDict()


class OneHot:
    def __init__(self, n_classes, to_float: bool = False):
        self.n_classes = n_classes
        self.to_float = to_float

    def __call__(self, label: torch.Tensor):
        return one_hot(label, self.n_classes).float() if self.to_float else one_hot(label, self.n_classes)


class DataToTensor:
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = torch.float
        self.dtype = dtype

    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)


class TSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# path = CIFAR10
def init_datasets():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                           OneHot(CLASSES, to_float=True)])
    global dataset
    dataset = CIFAR10(root=join(datasets, "CIFAR10"), train=True, download=True, transform=transform,
                      target_transform=target_transform)


def init_samplers():
    hetero_dir_part = CIFAR10Partitioner(dataset.targets,
                                         num_clients,
                                         balance=None,
                                         partition="dirichlet",
                                         dir_alpha=0.3,
                                         seed=2022)
    global samplers_dict
    for i in range(num_clients):
        samplers_dict[i] = TSampler(hetero_dir_part[i])


def init_data_loaders():
    global loaders_dict
    for i in range(num_clients):
        loaders_dict[i] = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     sampler=samplers_dict[i], num_workers=8,
                                     pin_memory=True)


def get_ranks():
    global hranks_dict
    for i in range(num_clients):
        hranks_dict[i] = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
        hranks_dict[i].wrapper.load_checkpoint(VGG16)
    global ranks_dict
    for i in range(num_clients):
        hranks_dict[i].get_rank(loaders_dict[i])
        ranks_dict[i] = hranks_dict[i].rank_dict


def store_ranks():
    global ranks_dict
    with open('ranks.ret', 'wb') as f:
        pickle.dump(ranks_dict, f)


def main():
    init_datasets()
    init_samplers()
    init_data_loaders()
    get_ranks()
    store_ranks()
