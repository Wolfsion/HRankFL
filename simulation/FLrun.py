from abc import ABC, abstractclassmethod
import os
import torch

from bases.fl.simulation.adaptive import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from control.algorithm import ControlModule
from bases.vision.sampler import FLSampler
from bases.nn.models.vgg import VGG11
from configs.cifar10 import *
import configs.cifar10 as config

from utils.save_load import mkdir_save

from control.argParser import PruningFLParser
from control.preEnv import *
from simulation.FLnodes import *
from data.samplers import CF10NIIDSampler
from data.dataProvider import get_data_loader
from model.modelUtil import vgg_16_bn

# deepcopy generate model
class AdaptiveFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.use_ip = args.initial_pruning
        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, args.max_density
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []
        self.start_adj_round = None

    def main(self):
        len_pre_rounds = 0
        if self.use_ip:
            print("Starting initial pruning stage...")
            # initial_pruning
            len_pre_rounds = self.server.initial_pruning(self.list_est_time, self.list_loss, self.list_acc,
                                                         self.list_model_size)
            print("Clients loading server model...")
            
            # load parameters and masks belong to conv2d & linear-fc
            for client in self.client_list:
                client.load_state_dict(self.server.model.state_dict())
                client.load_mask([layer.mask for layer in self.server.model.prunable_layers])

        print("Starting further pruning stage...")
        start = timer()
        for idx in range(self.max_round):
            # list for communcation between server and client 
            list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []
            is_adj_round = False

            # DISP: display maybe do valid or test
            if idx % self.config.EVAL_DISP_INTERVAL == 0:
                is_adj_round = self.check_adj_round(len_pre_rounds, idx)

            # every single client train and return sub result
            # state dict, num of input, square grad, learning rate
            for client in self.client_list:
                sd, npc, grad, last_lr = client.main(is_adj_round)
                list_state_dict.append(sd)
                list_num.append(npc)
                list_accum_sgrad.append(grad)
                list_last_lr.append(last_lr)

            # why check learning rate to be the same?
            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            # get desity_limit of current round 
            density_limit = None
            if self.max_d is not None:
                density_limit = self.max_d
            if self.tgt_d is not None:
                assert self.tgt_d <= self.max_d
                density_limit += (self.tgt_d - self.max_d) / self.max_round * idx

            # merge sub result and return entire mask and parameters
            # client provide: round, state dict, num of input, learning rate, square grad
            # main process: start timer, density limit, adj round(bool)
            # ret container: loss, acc, estimate time, model size
            list_mask, new_list_sd = self.server.main(idx, list_state_dict, list_num, last_lr, list_accum_sgrad, start,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size, is_adj_round, density_limit)

            # every client updata the merged parameters and mask
            # state dict, masks                                        
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                client.load_mask(list_mask)

if __name__ == "__main__":
    parser = PruningFLParser()
    args = parser.parse()

    torch.manual_seed(args.seed)

    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    # server
    server = CIFAR10AdaptiveServer(args, vgg_16_bn())
    list_models, list_indices = server.init_clients()

    # client 
    sa = CF10NIIDSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, 
                            args.client_selection, num_slices)
    GLOBAL_LOGGER.info("Sampler initialized")

    train_loader = get_data_loader(CIFAR10_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sa, num_workers=8, pin_memory=True)

    client_list = [CIFAR10AdaptiveClient(list_models[idx], config, args.use_adaptive) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    GLOBAL_LOGGER.info("All initialized. Experiment is {}. Use adaptive = {}. Use initial pruning = {}. Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "Target density = {}".format(CIFAR10_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
                                       num_users, args.seed, MAX_ROUND, args.target_density))

    fl_runner = AdaptiveFL(args, config, server, client_list)
    fl_runner.main()
