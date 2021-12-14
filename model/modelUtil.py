import torch
import torch.nn as nn
from torch.nn.modules.instancenorm import LazyInstanceNorm1d

from model import vgg
from control.preEnv import *

hasParameter = lambda x: len(list(x.parameters())) != 0

def traverse(model: nn.Module):
    list1 = []
    list2 = []
    traverse_module(model, hasParameter, list1, list2)
    prunable_nums = [ly_id for ly_id, ly in enumerate(list1) if not isinstance(ly, nn.BatchNorm2d)]
    list3 = [list1[ly_id] for ly_id in prunable_nums]
    list4 = [list2[ly_id] for ly_id in prunable_nums]
    ret = {
            "param_layers":list1, 
            "param_layer_prefixes":list2, 
            "prunable_layers":list3, 
            "prunable_layer_prefixes":list4
        }
    return ret

def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")

def vgg_16_bn(compress_rate):
    return vgg.VGG(compress_rate=compress_rate)

def model_device(model: nn.Module):
    curt = str(next(model.parameters()).device)
    if (len(curt) > CPU_STR_LEN):
        return GPU
    else:
        return CPU
    
