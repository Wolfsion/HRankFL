import argparse
from copy import deepcopy
from collections import OrderedDict
import torch
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import numpy as np

import data.samplers as samplers
from data.dataProvider import get_data_loader
import model.modelUtil as modelUtil
from model.vwrapper import VWrapper
from pruning.vhrank import VGG16HRank
from federal.FLbase import *
from control.pathHandler import *
from federal.FLnodes import *
from federal.Message import *
from control.preEnv import *
from control.runtimeEnv import *

sa = samplers.CF10NIIDSampler(num_slices, MAX_ROUND, data_per_client_epoch, 
                                True, client_per_round)
workers_lodder = get_data_loader(CIFAR10_NAME, data_type="train", 
                                batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                sampler=sa, num_workers=8, pin_memory=True)
GLOBAL_LOGGER.info("Sampler initialized")

# CIFAR10 VGG16 HRank
class CVHMaster(FLMaster):    
    def init_algorithm(self):
        model = modelUtil.vgg_16_bn(compress_rate)
        self.wrapper = VWrapper(model)
        self.ranks = None
        self.mdict = None
        self.prunning_rate = None
    
    def save_exp_config(self):
        exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
                            "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
                            "lrhl": LR_HALF_LIFE}
        args_config = vars(self.args)
        configs = exp_const_config.copy()
        configs.update(args_config)
        modelUtil.mkdir_save(configs, file_repo.configs('exp_config.snap'))

    def distribute_dict():
        pass

    def distribute_ndarray():
        pass

class CVHWorker(FLWorker):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.alg_obj.model.train()

    def init_loader(self):
        self.loader = workers_lodder

    def init_algorithm(self):
        model = modelUtil.vgg_16_bn(compress_rate=[0.]*100)
        self.alg_obj = VGG16HRank(model)
        self.alg_obj.get_rank(self.loader)

# deepcopy generate model
class CVHRun():
    def __init__(self, args: argparse.Namespace) -> None:
        self.pipe = FLSimNet()
        self.master = CVHMaster(args)
        self.workers = [CVHWorker(args) for _ in range(workers)]

    def distribute_model_dict(master: CVHMaster):
        pass

    def updata_model_dict(worker: CVHWorker):
        pass

    def merge_model_dict(master: CVHMaster):
        pass

    def merge_rank(master: CVHMaster):
        pass
