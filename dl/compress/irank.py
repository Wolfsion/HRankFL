import torch
import torch.nn as nn
import torch.utils.data as tdata
import numpy as np
from collections import OrderedDict
from typing import List

from env.preEnv import *
from env.runtimeEnv import *

from dl.compress.vhrank import HRank
from dl.model.vwrapper import VWrapper
from dl.model import modelUtil


class IterRank(HRank):
    def __init__(self, model: nn.Module, model_type: int = 1) -> None:
        super().__init__(model_type, model)
        self.wrapper = VWrapper(model)
        self.model = self.wrapper.model

        self.cp_model = None
        self.cp_model_sd = None

        self.relu_cfg = self.wrapper.device.access_model().relucfg
        self.rank_dict = OrderedDict()

    def device_train(self, loader: tdata.DataLoader, rounds: int):
        for i in range(rounds):
            self.learn_run(loader)

    def learn_run(self, loader: tdata.DataLoader):
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= train_limit:
                break
            loss, cort = self.wrapper.step_eva(inputs, targets, train=True)
            test_loss += loss
            correct += cort
            total += targets.size(0)
            GLOBAL_LOGGER.info('Train:batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def feed_run(self, loader: tdata.DataLoader):
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if batch_idx >= limit:
                    break

                loss, cort = self.wrapper.step_eva(inputs, targets)

                test_loss += loss
                correct += cort
                total += targets.size(0)

                GLOBAL_LOGGER.info('batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def init_cp_model(self, pruning_rate: List[float]):
        self.cp_model = modelUtil.vgg_16_bn(pruning_rate)
        self.cp_model_sd = self.cp_model.state_dict()

    def get_rank(self, loader: tdata.DataLoader):
        for cov_id in self.relu_cfg:
            cov_layer = self.wrapper.device.access_model().features[cov_id]
            handler = cov_layer.register_forward_hook(self.get_feature_hook)
            self.feed_run(loader)
            handler.remove()

            self.rank_dict[cov_id] = self.feature_result.numpy()
            np.save(file_repo.rank(), self.feature_result.numpy())

            self.feature_result = torch.tensor(0.)
            self.total = torch.tensor(0.)
        file_repo.reset_rank_index()


    def deserialize_rank(self):
        for cov_id in enumerate(self.relu_cfg):
            GLOBAL_LOGGER.info('loading rank from: ' + file_repo.rank_dir())
            rank = np.load(file_repo.rank())
            self.rank_dict[cov_id] = rank
        file_repo.reset_rank_index()

    def load_params(self, rank_dict: OrderedDict = None):
        last_select_index = None  # Conv index selected in the previous layer
        if rank_dict is None:
            if not self.rank_dict:
                self.deserialize_rank()
            iter_ranks = iter(self.rank_dict.values())
        else:
            iter_ranks = iter(rank_dict.values())
        osd = self.model.module.state_dict()

        for name, module in self.model.named_modules():
            if self.wrapper.device.GPUs:
                name = name.replace('module.', '')

            if isinstance(module, nn.Conv2d):

                ori_weight = osd[name + '.weight']
                cur_weight = self.cp_model_sd[name + '.weight']
                ori_filter_num = ori_weight.size(0)
                cur_filter_num = cur_weight.size(0)

                if ori_filter_num != cur_filter_num:

                    rank = next(iter_ranks)
                    # preserved filter index based on rank
                    select_index = np.argsort(rank)[ori_filter_num - cur_filter_num:]

                    # traverse list in increase order(not necessary step)
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                self.cp_model_sd[name + '.weight'][index_i][index_j] = \
                                    osd[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            self.cp_model_sd[name + '.weight'][index_i] = \
                                osd[name + '.weight'][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for i in range(ori_filter_num):
                        for index_j, j in enumerate(last_select_index):
                            self.cp_model_sd[name + '.weight'][i][index_j] = \
                                osd[name + '.weight'][i][j]

                # retain origin channel
                else:
                    self.cp_model_sd[name + '.weight'] = ori_weight
                    last_select_index = None

        self.cp_model.load_state_dict(self.cp_model_sd)
