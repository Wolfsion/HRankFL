from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata

from env.preEnv import *
from env.runtimeEnv import *
from dl.model.vwrapper import VWrapper
from dl.model import modelUtil


# deserialize rank file

class HRank(ABC):
    ERROR_MESS1 = "Illegal model_type."
    ERROR_MESS2 = "Loader is none, must use random data."
    feature_result: torch.Tensor = torch.tensor(0.)
    total: torch.Tensor = torch.tensor(0.)

    def __init__(self, model_type: ModelType, model: nn.Module) -> None:
        self.mtype = model_type.value
        self.wrapper = VWrapper(model)
        self.model = self.wrapper.model
        self.curt_dict = None
        self.curt_inputs = None
        self.rank_dict = OrderedDict()
        self.cp_model = None
        self.cp_model_sd = None
        self.all_batch = 0

        self.random_batch_size = 32
        self.random_channel = 3
        self.random_labels = 10

    # get feature map of certain layer via hook
    def get_feature_hook(self, module, input, output):
        imgs = output.shape[0]
        channels = output.shape[1]
        ranks = torch.tensor([torch.linalg.matrix_rank(output[i, j, :, :]).item()
                              for i in range(imgs) for j in range(channels)])
        ranks = ranks.view(imgs, -1).float()

        # merge channel rank of all imgs
        ranks = ranks.sum(0)

        self.feature_result = self.feature_result * self.total + ranks
        self.total = self.total + imgs
        self.feature_result = self.feature_result / self.total

    def drive_hook(self, sub_module: nn.Module, loader: tdata.DataLoader,
                   hook_id: int = None, random: bool = False):
        handler = sub_module.register_forward_hook(self.get_feature_hook)
        if random:
            self.feed_random_run()
        else:
            self.feed_run(loader)
        handler.remove()

        if hook_id is not None:
            self.rank_dict[hook_id] = self.feature_result.numpy()

        np.save(file_repo.rank(), self.feature_result.numpy())
        self.feature_result = torch.tensor(0.)
        self.total = torch.tensor(0.)

    def learn_run(self, loader: tdata.DataLoader):
        test_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        while True:
            try:
                self.all_batch += 1
                inputs, targets = next(iter(loader))

                # # single train config
                # if batch_idx >= train_limit:
                #     self.valid_performance(loader)
                #     self.interrupt_disk()
                #     break

                # union train config
                if batch_idx >= union_train_limit:
                    # self.wrapper.valid_performance(loader)
                    break

                loss, cort = self.wrapper.step(inputs, targets, train=True)
                test_loss += loss
                correct += cort
                total += targets.size(0)
                GLOBAL_LOGGER.info('Train:batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                batch_idx += 1

            except StopIteration:
                self.valid_performance(loader)
                self.interrupt_disk()
                GLOBAL_LOGGER.info('The loader is over.')
                break
        # for batch_idx, (inputs, targets) in enumerate(loader):
        #     if batch_idx >= union_train_limit:
        #         break
        #
        #     loss, cort = self.wrapper.step(inputs, targets, train=True)
        #
        #     test_loss += loss
        #     correct += cort
        #     total += targets.size(0)
        #     GLOBAL_LOGGER.info("batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
        #         batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.curt_dict = self.wrapper.model.state_dict()
        self.curt_inputs = total

    def feed_run(self, loader: tdata.DataLoader):
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if batch_idx >= limit:
                    break
                loss, cort = self.wrapper.step(inputs, targets)
                test_loss += loss
                correct += cort
                total += targets.size(0)

                GLOBAL_LOGGER.info('Test batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def feed_random_run(self):
        with torch.no_grad():
            for batch_idx in range(limit):
                GLOBAL_LOGGER.info('using random data...')
                inputs = torch.randn(self.random_batch_size, 3, 32, 32)
                targets = torch.randn(self.random_batch_size, self.random_labels)
                self.wrapper.step(inputs, targets)

    def interrupt_disk(self, name: str = None):
        if name is None:
            self.wrapper.save_checkpoint(file_repo.configs('exp_config.snap'))
        else:
            self.wrapper.save_checkpoint(file_repo.configs(name))

    def restore_disk(self, path: str = None):
        if path is None:
            self.wrapper.load_checkpoint(file_repo.configs('exp_config.snap'))
        else:
            self.wrapper.load_checkpoint(file_repo.configs(path))

    def interrupt_mem(self) -> dict:
        return self.wrapper.model.state_dict()

    def restore_mem(self, state_dict: dict):
        self.wrapper.model.load_state_dict(state_dict)

    def valid_performance(self, loader: tdata.DataLoader):
        wrapper = VWrapper(self.cp_model)
        wrapper.valid_performance(loader)

    @abstractmethod
    def get_rank(self, loader: tdata.DataLoader):
        pass

    @abstractmethod
    def deserialize_rank(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    @abstractmethod
    def init_cp_model(self, pruning_rate: List[float]):
        pass


class VGG16HRank(HRank):
    def __init__(self, model: nn.Module, model_type: ModelType = ModelType.VGG) -> None:
        super().__init__(model_type, model)

        self.relu_cfg = self.wrapper.device.access_model().relucfg
        self.cov_order = 0


    def get_rank(self, loader: tdata.DataLoader = None, random: bool = False):
        if loader is None:
            assert random, self.ERROR_MESS1
        for cov_id in self.relu_cfg:
            cov_layer = self.wrapper.device.access_model().features[cov_id]
            self.drive_hook(cov_layer, loader, self.cov_order, random)
            self.cov_order += 1
        file_repo.reset_rank_index()
        self.cov_order = 0

    def get_rank_plus(self):
        pass

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
        osd = self.model.state_dict()

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

    def init_cp_model(self, pruning_rate: List[float]):
        self.cp_model = modelUtil.vgg_16_bn(pruning_rate)
        self.cp_model_sd = self.cp_model.state_dict()


class ResNet56HRank(HRank):

    def __init__(self, model: nn.Module, model_type: ModelType = ModelType.RESNET) -> None:
        super().__init__(model_type, model)

    def get_rank(self, loader: tdata.DataLoader):
        hook_cnt = 0
        cov_layer = self.wrapper.model.relu
        self.drive_hook(cov_layer, loader, hook_cnt)
        hook_cnt += 1

        # ResNet56 per block
        for i in range(3):
            # eval()!!!
            block = eval('self.wrapper.model.layer%d' % (i + 1))
            for j in range(9):
                for _relu in range(2):
                    if _relu == 0:
                        cov_layer = block[j].relu1
                    else:
                        cov_layer = block[j].relu2
                    self.drive_hook(cov_layer, loader, hook_cnt)
                    hook_cnt += 1
        file_repo.reset_rank_index()

    def deserialize_rank(self):
        file_repo.reset_rank_index()

    def init_cp_model(self, pruning_rate: List[float]):
        pass

    def load_params(self):
        pass


class ResNet50HRank(HRank):
    pass
