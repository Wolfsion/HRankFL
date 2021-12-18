from typing import List
import torch
from torch import nn
import logging

from control import vlogger


# 设备、模型、张量动态绑定
# SingleTon
# path路径由底层直接维护
class VDevice():
    CPU_STR_LEN = 3
    LOGGER_PATH = "logs/device_info.log"
    ERROR_MESS1 = "GPU is not available."

    def __init__(self, gpu: True, ids: List = [0], logger: logging.Logger = None) -> None:
        self.flag = torch.cuda.is_available()
        self.dev_list = ids
        if (logger == None):
            self.logger = vlogger.VLogger(self.LOGGER_PATH).logger
        if gpu:
            assert self.flag == True, self.ERROR_MESS1
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True
        else:
            self.device = torch.device("cpu")
            self.last_choice = False
    
    def device_switch(self):
        if self.last_choice:
            self.device = torch.device("cpu")
            self.last_choice = False
        else:
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True

    def to_gpu(self, model: torch.nn.Module):
        if len(self.dev_list) > 1:
            nn.DataParallel(model, device_ids=self.dev_list).cuda()
            model.to(self.device)
        else:
            model.to(self.device)
    
    def to_cpu(self, model: torch.nn.Module):
        model.cpu()

    def tensor_switch(self, tensor: torch.Tensor):
        curt = str(tensor.device)
        if len(curt) > self.CPU_STR_LEN:
            tensor.cpu()
            self.logger.info("tensor will load on cpu.")
        else:
            tensor.to(self.device)
            self.logger.info("tensor will load on gpu.")

    def model_switch(self, model: torch.nn.Module):
        curt = str(next(model.parameters()).device)
        if len(curt) > self.CPU_STR_LEN:
            self.to_cpu(model)
            self.logger.info("model will use cpu.")
        else:
            self.to_gpu(model)
            self.logger.info("model will use gpu.")
 
# Auto config
class VADevice():
    CPU_STR_LEN = 3
    MODLE_PATH = "milestone"
    LOGGER_PATH = "logs/device_info.log"
    ERROR_MESS1 = "GPU is not available."
    ERROR_MESS2 = "Model must be not null."

    def __init__(self, gpu: True, ids: List = [0], logger: logging.Logger = None) -> None:
        self.flag = torch.cuda.is_available()
        self.dev_list = ids
        self.model = None
        self.GPUs = False
        if (logger == None):
            self.logger = vlogger.VLogger(self.LOGGER_PATH).logger
        if gpu:
            assert self.flag == True, self.ERROR_MESS1
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True
        else:
            self.device = torch.device("cpu")
            self.last_choice = False
    
    def bind_model(self, model: nn.Module):
        self.model = model
        if self.last_choice:
            self.to_gpu(self.model)
        else:
            self.to_cpu(self.model)

    def switch_device(self):
        if self.last_choice:
            self.device = torch.device("cpu")
            self.last_choice = False
        else:
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True

    def to_gpu(self):
        assert self.model != None, self.ERROR_MESS2
        if len(self.dev_list) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.dev_list)
            self.model.to(self.device)
            self.GPUs = True
        else:
            self.model.to(self.device)
            self.GPUs = False
        if self.last_choice == False:
            self.switch_device()
    
    def to_cpu(self):
        assert self.model != None, self.ERROR_MESS2
        self.model.cpu()
        self.GPUs = False
        if self.last_choice:
            self.switch_device()

    def on_tensor(self, *tensors: torch.Tensor):
        for t in tensors:
            yield t.to(self.device)

    def save_model(self, path):
        assert self.model != None, self.ERROR_MESS2
        if self.dev_list > 1:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        assert self.model != None, self.ERROR_MESS2
        self.model.load_state_dict(torch.load(path))

    def direct_load_model():
        pass

    def direct_save_model():
        pass

    def save_checkpoint():
        pass
    def load_checkpoint():
        pass
