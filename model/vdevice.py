from typing import List
import torch
from torch import nn
import logging

from control import vlogger
class VDevice():
    CPU_STR_LEN = 3
    LOGGER_PATH = "logs/device_info.log"
    ERROR_MESS1 = "GPU is not available."
    ERROR_MESS2 = "Do not use gpu."

    def __init__(self, gpu: True, ids: List = [0], logger: logging.Logger = None) -> None:
        self.flag = torch.cuda.is_available()
        if (logger == None):
            self.logger = vlogger.VLogger(self.LOGGER_PATH).logger
        if gpu:
            assert self.flag == True, self.ERROR_MESS1
            self.device = torch.device("cuda:%d" % ids[0])
            self.user_choice = True
            self.dList = ids
        else:
            self.device = torch.device("cpu")
            self.user_choice = False
    
    def to_gpu(self, model: torch.nn.Module):
        assert self.user_choice == True, self.ERROR_MESS2
        if len(self.dList) > 1:
            nn.DataParallel(model, device_ids=self.dList).cuda()
            model.to(self.device)
        else:
            model.to(self.device)
    
    def to_cpu(self, model: torch.nn.Module):
        model.cpu()

    def tensor_switch(self, tensor: torch.tensor):
        curt = str(tensor.device)
        if len(curt) > self.CPU_STR_LEN:
            tensor.cpu()
            self.logger.info("tensor will load on cpu.")
        else:
            assert self.user_choice == True, self.ERROR_MESS2
            tensor.gpu()
            self.logger.info("tensor will load on gpu.")

    def model_switch(self, model: torch.nn.Module):
        curt = str(next(model.parameters()).device)
        if len(curt) > self.CPU_STR_LEN:
            self.to_cpu(model)
            self.logger.info("model will use cpu.")
        else:
            assert self.user_choice == True, self.ERROR_MESS2
            self.to_cpu(model)
            self.logger.info("model will use cpu.")
   