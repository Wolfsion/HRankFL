import os
import pickle
import warnings
import torch
import torch.nn as nn
import torch.utils.data as tdata
from thop import profile
from timeit import default_timer as timer
from copy import deepcopy

from dl.model import vgg
from env.preEnv import *
from env.runtimeEnv import *
from dl.model.vwrapper import VWrapper

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
        opened_f.close()
    return obj

def valid_performance(loader: tdata.DataLoader, wrapper: VWrapper):
    first_feed = True
    flops = 0
    params = 0
    test_loss = 0
    correct = 0
    total = 0

    time_start = timer()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if first_feed:
                tmp = deepcopy(wrapper.model.module)
                flops, params = profile(tmp.cpu(), inputs=(inputs,))
                first_feed = False

            if batch_idx >= valid_limit:
                break

            loss, cort = wrapper.step_eva(inputs, targets)
            test_loss += loss
            correct += cort

            total += targets.size(0)

    time_cost = timer() - time_start
    total_params = sum(p.numel() for p in wrapper.model.parameters())
    total_trainable_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)

    GLOBAL_LOGGER.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (test_loss / valid_limit, 100. * correct / total, correct, total))

    GLOBAL_LOGGER.info('Time cost: %.3f | FLOPs: %d | Params: %d'
                       % (time_cost, flops, params))

    GLOBAL_LOGGER.info('Total params: %d | Trainable params: %d'
                       % (total_params, total_trainable_params))
