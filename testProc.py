from argparse import Namespace
import argparse
import time

from control.preEnv import DataSetType
from data import samplers
from data.dataProvider import *
from control import *

import torch
import torchvision

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

from abc import ABC, abstractmethod

from model import modelUtil
from model import vdevice
from pruning.vhrank import VGG16HRank

num_clients = 100
num_classes = 10
seed = 1
hist_color = '#4169E1'

EXP_NAME = "CIFAR10"
CLIENT_BATCH_SIZE = 20

def testTensor():
    a = torch.tensor([[1,2,3,4],[4,4,4,4]])
    print(a.sum(0))

def testModel():
    model = modelUtil.vgg_16_bn(compress_rate=[0.]*100)
    hrank = VGG16HRank(1, model)
    sa = samplers.CF10NIIDSampler(1,100,10001,100,True,10)
    loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                sampler=sa, num_workers=8, pin_memory=True)
    hrank.get_rank(loader)
        
    print("----------------------")
        
def testData():
    sa = samplers.CF10NIIDSampler(1,100,10001,100,True,10)
    
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sa, num_workers=2)
    for (inputs, targets) in train_loader:
        print(inputs.size())
    print("----------------------")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num',
                        help="Experiment name",
                        action='store',
                        dest='get',
                        type=int,
                        required=True)
    return parser.parse_args()

class A():
    def __init__(self, obj) -> None:
        self.obj = obj

class B():
    def __init__(self, obj) -> None:
        self.a = A(obj)
        self.obj = obj
    

if __name__ == "__main__":
    exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
                            "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
                            "lrhl": LR_HALF_LIFE}
    modelUtil.mkdir_save(exp_const_config, 'test.pt')
    print(modelUtil.pickle_load('test.pt'))
    print("----------------------")