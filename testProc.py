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
from pruning.hrank import VGG16HRank

num_clients = 100
num_classes = 10
seed = 1
hist_color = '#4169E1'

EXP_NAME = "CIFAR10"
CLIENT_BATCH_SIZE = 20

def test(*num):
    for i in num:
        yield i+1

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
    print("----------------------")

class A():
    def __init__(self) -> None:
        self.a = 1

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    testModel()
    print("----------------------")