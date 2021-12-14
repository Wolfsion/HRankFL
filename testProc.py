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
    all = len(model.features)
    relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
    for cov_id in relucfg:
        print(model.features[cov_id])
        
    print("----------------------")
        
def testData():
    sa = samplers.CF10NIIDSampler(1,100,10001,100,True,10)
    
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sa, num_workers=2)
    print("----------------------")

if __name__ == "__main__":
    testTensor()
    print("----------------------")