import os
import pickle
import warnings
import torch
import torch.nn as nn
import torch.utils.data as tdata
from thop import profile
from timeit import default_timer as timer
from copy import deepcopy

from dl.model.mobilenet import MobileNetV2
from dl.model.resnet import ResNet, BasicBlock
from dl.model.vgg import VGG16
from dl.model.vgg import VGG11
from env.preEnv import *
from env.runtimeEnv import *


def initialize(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# for cifar10
def vgg_16_bn(compress_rate):
    return VGG16(compress_rate=compress_rate)


def vgg_11_bn(compress_rate):
    return VGG11(compress_rate=compress_rate)


# for cifar100
def resnet_56(compress_rate):
    return ResNet(BasicBlock, 56, compress_rate=compress_rate, num_classes=100)


# for cifar10
def resnet_110(compress_rate):
    return ResNet(BasicBlock, 110, compress_rate=compress_rate)


# for cifar100
def mobilenet_v2(compress_rate):
    return MobileNetV2(compress_rate=compress_rate, width_mult=1)


def model_device(model: nn.Module):
    curt = str(next(model.parameters()).device)
    if (len(curt) > CPU_STR_LEN):
        return GPU
    else:
        return CPU


# load obj not only model: nn.Moudle
def mkdir_save(obj, f):
    dir_name = os.path.dirname(f)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    # disabling warnings from torch.Tensor's reduce function. See issue: https://github.com/pytorch/pytorch/issues/38597
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(f, "wb") as opened_f:
            pickle.dump(obj, opened_f)
            opened_f.close()


def pickle_load(f):
    with open(f, "rb") as opened_f:
        obj = pickle.load(opened_f)
    return obj


def dict_diff(dict1: dict, dict2: dict):
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        if k1 != k2:
            GLOBAL_LOGGER.info('Key beq:dict1_key:', k1)
            GLOBAL_LOGGER.info('Key beq:dict2_key:', k2)
        else:
            if not v1.equal(v2):
                GLOBAL_LOGGER.info(f"The value of key:{k1} is not equal.")
