from control.dsets import DataSetType
from data import samplers
from data.dataProvider import *
from control import *

import torch
import torchvision

from abc import ABC, abstractmethod

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

num_clients = 100
num_classes = 10
seed = 1
hist_color = '#4169E1'

EXP_NAME = "CIFAR10"
CLIENT_BATCH_SIZE = 20
        
def testData():
    sa = samplers.CF10NIIDSampler(1,100,10001,100,True,10)
    
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sa, num_workers=2)
    print("----------------------")

if __name__ == "__main__":
    testData()
    print("----------------------")