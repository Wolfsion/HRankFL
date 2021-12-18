from abc import ABC, abstractmethod
from collections import OrderedDict
from torch import Tensor
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as tdata
import collections

from control.preEnv import *
import control.runtimeEnv as args
from model.vwrapper import VWrapper

class HRank(ABC):
    ERROR_MESS1 = "illegal mType(int)"
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    def __init__(self, mType: int, model: nn.Module) -> None:
        assert mType > ModelType.LOWWER.value, self.ERROR_MESS1
        assert mType < ModelType.UPPER.value, self.ERROR_MESS1
        self.mType = mType
        self.model = model

    #get feature map of certain layer via hook
    def get_feature_hook(self, module, input, output):
        imgs = output.shape[0]
        channels = output.shape[1]
        ranks = torch.tensor([torch.linalg.matrix_rank(output[i,j,:,:]).item() 
                            for i in range(imgs) for j in range(channels)])
        ranks = ranks.view(imgs, -1).float()
    
        # merge channel rank of all imgs
        ranks = ranks.sum(0)

        self.feature_result = self.feature_result * self.total + ranks
        self.total = self.total + imgs
        self.feature_result = self.feature_result / self.total

    @abstractmethod
    def feed_run(self, loader: tdata.DataLoader):
        pass

    @abstractmethod
    def get_rank(self, loader: tdata.DataLoader):
        pass

    @abstractmethod
    def get_mask(self):
        pass

    @abstractmethod
    def fix_structure(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    @abstractmethod
    def upload_comm_dict(self):
        pass

    @abstractmethod
    def download_comm_dict(self):
        pass

class VGG16HRank(HRank):
    def __init__(self, model: nn.Module, rank_path: str, mType: int = 1) -> None:
        super().__init__(mType, model)
        self.wrapper = VWrapper(model)
        self.curt_sd = model.state_dict()
        self.rp = rank_path

    def get_rank(self, loader):
        if len(args.gpu) > 1:
            relucfg = self.model.module.relucfg
        else:
            relucfg = self.model.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = self.model.features[cov_id]
            handler = cov_layer.register_forward_hook(self.get_feature_hook)
            self.feed_run(loader)
            handler.remove()

            if not os.path.isdir('ranks/'+args.arch+'_limit%d'%(args.limit)):
                os.mkdir('ranks/'+args.arch+'_limit%d'%(args.limit))
            np.save('ranks/'+args.arch+'_limit%d'%(args.limit)+'/ranks' + str(i + 1) + '.npy', self.feature_result.numpy())

            self.feature_result = torch.tensor(0.)
            self.total = torch.tensor(0.)
    
    def load_params(self, ori_state_dict: 'OrderedDict[str, Tensor]'):
        last_select_index = None #Conv index selected in the previous layer
        conv_cnt = 0
        osd = ori_state_dict

        if self.wrapper.device.GPUs:
            name_base='module.'
        else:
            name_base=''

        prefix = args.rank_conv_prefix+'/rank_conv'
        subfix = ".npy"

        for name, module in self.model.named_modules():
            if self.wrapper.device.GPUs:
                name = name.replace('module.', '')

            if isinstance(module, nn.Conv2d):
                conv_cnt += 1
                ori_weight = osd[name + '.weight']
                cur_weight = self.curt_sd[name_base + name + '.weight']
                ori_filter_num = ori_weight.size(0)
                cur_filter_num = cur_weight.size(0)

                if ori_filter_num != cur_filter_num:
                    cov_id = conv_cnt
                    GLOBAL_LOGGER.info('loading rank from: ' + prefix + str(cov_id) + subfix)

                    rank = np.load(prefix + str(cov_id) + subfix)

                    select_index = np.argsort(rank)[ori_filter_num-cur_filter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                self.curt_sd[name_base + name + '.weight'][index_i][index_j] = \
                                    osd[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            self.curt_sd[name_base + name + '.weight'][index_i] = \
                                    osd[name + '.weight'][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for i in range(ori_filter_num):
                        for index_j, j in enumerate(last_select_index):
                            self.curt_sd[name_base + name + '.weight'][i][index_j] = \
                                osd[name + '.weight'][i][j]
                
                # retain origin channel
                else:
                    self.curt_sd[name_base + name + '.weight'] = ori_weight
                    last_select_index = None

        self.model.load_state_dict(self.curt_sd)

    def feed_run(self, loader: tdata.DataLoader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = args.limit

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                #use the first 5 batches to estimate the rank.
                if batch_idx >= limit:
                    break

                loss, cort = self.wrapper.step_eva(inputs, targets)
                
                test_loss += loss
                correct += cort
                total += targets.size(0)
                
                GLOBAL_LOGGER.info('batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def upload_comm_dict(self) -> OrderedDict:
        upload = collections.OrderedDict()
        # after call self.load_params()
        upload['state_dict'] = self.curt_sd
        return upload

    def download_comm_dict(self) -> OrderedDict:
        download = collections.OrderedDict()
        # after call server.avgParams()
        download['state_dict'] = self.model.state_dict()
        return download

    pipeDict = None
    def send_dict(self, od:OrderedDict) -> None:
        self.pipeDict = od

    def receive(self) -> OrderedDict:
        return self.pipeDict