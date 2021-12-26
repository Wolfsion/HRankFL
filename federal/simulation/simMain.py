from abc import ABC, abstractclassmethod
import os
import torch
from torch.optim import lr_scheduler
from collections import OrderedDict

from control.argParser import PruningFLParser
from control.preEnv import *
from federal.FLnodes import *
from data.samplers import CF10NIIDSampler
from data.dataProvider import get_data_loader
from federal.simulation.CVH import CVHRun
from model.modelUtil import vgg_16_bn

def main():
    # parser = PruningFLParser()
    # args = parser.parse()
    # GLOBAL_LOGGER.info("All initialized. Experiment is {}. Client selection = {}. "
    #       "Num users = {}. Seed = {}. Max round = {}. "
    #       "Target density = {}".format(CIFAR10_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
    #                                    workers, args.seed, MAX_ROUND, args.target_density))
    dic = {"a":1, "b":2}
    args = argparse.Namespace(**dic)
    fl_runner = CVHRun(args)
    fl_runner.download_dict()
    fl_runner.download_cp_rate()
    fl_runner.upload_download_ranks()

    
