import argparse
import os
from copy import deepcopy
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from data.dataProvider import DataLoader

from simulation.FLbase import *
from control.preEnv import *
from control.pathHandler import *
import data.samplers as samplers
from data.dataProvider import get_data_loader
import model.modelUtil as modelUtil
from pruning.vhrank import VGG16HRank

class FLServer(ABC):
    def __init__(self, args: argparse.Namespace, model: nn.Module, save_interval=50):
        self.args = args
        self.save_interval = save_interval

        self.loader: DataLoader = None
        self.init_loader()
        self.init_algorithm()
        self.save_runtime_config()

    @abstractmethod
    def init_loader(self):
        pass

    @abstractmethod
    def init_algorithm(self):
        pass

    @abstractmethod
    def save_runtime_config(self):
        pass

    @abstractmethod
    def receive_merge(self):
        pass

    @abstractmethod
    def send_distribute(self):
        pass

class FLClient(ABC):
    def __init__(self, model):
        self.loader: DataLoader = None
        self.init_loader()
        self.init_algorithm()

    @abstractmethod
    def init_loader(self):
        pass

    def init_algorithm(self):
        pass
    
    @abstractmethod
    def receive_update(self):
        pass

    @abstractmethod
    def send_submit(self):
        pass
    
    @abstractmethod
    def run_train(self):
        pass

# CIFAR10 VGG16 HRank
class CVHServer(FLServer):
    def __init__(self, args: argparse.Namespace, save_interval=50):
        self.file_repo = HRankPathGather(VGG_MODLE_PATH, DATASETS_PATH, RANKS_PATH)
        super().__init__(args, save_interval=save_interval)
        self.alg_obj.model.eval()

    # replace loader config
    def init_loader(self):
        num_slices = 100
        data_per_client_epoch = 100
        client_per_round = 10
        sa = samplers.CF10NIIDSampler(num_slices, MAX_ROUND, data_per_client_epoch, 
                                        True, client_per_round)
        self.loader = get_data_loader(CIFAR10_NAME, data_type="train", 
                        batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                        sampler=sa, num_workers=8, pin_memory=True)

    # replace compress_rate
    def init_algorithm(self):
        model = modelUtil.vgg_16_bn(compress_rate=[0.]*100)
        self.alg_obj = VGG16HRank(model)
        self.alg_obj.get_rank(self.loader)
    
    def save_exp_config(self):
        exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
                            "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
                            "lrhl": LR_HALF_LIFE}
        args_config = vars(self.args)
        configs = dict(exp_const_config.items() + args_config.items())
        modelUtil.mkdir_save(configs, self.file_repo.configs('exp_config.snap'))

    def main(self, idx, list_sd, list_num_proc, lr, list_accumulated_sgrad, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, density_limit=None):
        total_num_proc = sum(list_num_proc)

        # merge parameters: Avg with weight(proportion of data)
        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_sd):
                    if key in state_dict.keys():
                        mask = self.model.get_mask_by_name(key)
                        if mask is None:
                            inc_val = state_dict[key] - param
                        else:
                            inc_val = state_dict[key] - param * self.model.get_mask_by_name(key)

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        # DISP:display maybe test or valid
        if idx % self.config.EVAL_DISP_INTERVAL == 0:
            loss, acc = self.model.evaluate(self.test_loader)
            list_loss.append(loss)
            list_acc.append(acc)

            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Loss/acc (at round #{}) = {}/{}".format((len(list_loss) - 1) * self.config.EVAL_DISP_INTERVAL, loss,
                                                           acc))
            print("Estimated time = {}".format(sum(list_est_time)))
            print("Elapsed time = {}".format(timer() - start))
            print("Current lr = {}".format(lr))


        '''
            server pruning
        '''

        # adj:adjust adjust(decrease or add) mask to do with network structure
        if self.use_adaptive and is_adj_round:
            alg_start = timer()

            for d in list_accumulated_sgrad:
                for k, sg in d.items():
                    self.control.accumulate(k, sg)

            print("Running adaptive pruning algorithm")
            max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))
            self.control.adjust(max_dec_diff, max_density=density_limit)
            print("Total alg time = {}. Max density = {}.".format(timer() - alg_start, density_limit))
            print("Num params:")

            # parameters proportion used
            disp_num_params(self.model)

        '''
            server pruning
        '''
        
        # time cost:time const * num of weight 1
        est_time = self.config.TIME_CONSTANT
        for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
            est_time += layer.num_weight * (comp_coeff + self.config.COMM_COEFFICIENT)

        # model_size depend on num of weight 1
        model_size = self.model.calc_num_all_active_params(True)
        list_est_time.append(est_time)
        list_model_size.append(model_size)

        if idx % self.save_interval == 0:
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_acc, os.path.join(self.save_path, "accuracy.pt"))
            mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
            mkdir_save(list_model_size, os.path.join(self.save_path, "model_size.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

        # return mask and parameters
        return [layer.mask for layer in self.model.prunable_layers], [self.model.state_dict() for _ in
                                                                      range(self.config.NUM_CLIENTS)]

class CVHClient(FLClient):
    def __init__(self, args: argparse.Namespace, save_interval=50):
        self.file_repo = HRankPathGather(VGG_MODLE_PATH, DATASETS_PATH, RANKS_PATH)
        super().__init__(args, save_interval=save_interval)
        self.alg_obj.model.train()

    # replace loader config
    def init_loader(self):
        num_slices = 100
        data_per_client_epoch = 100
        client_per_round = 10
        sa = samplers.CF10NIIDSampler(num_slices, MAX_ROUND, data_per_client_epoch, 
                                        True, client_per_round)
        self.loader = get_data_loader(CIFAR10_NAME, data_type="train", 
                        batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                        sampler=sa, num_workers=8, pin_memory=True)

    def init_algorithm(self):
        model = modelUtil.vgg_16_bn(compress_rate=[0.]*100)
        self.alg_obj = VGG16HRank(model)
        self.alg_obj.get_rank(self.loader)

    def main(self, is_adj_round):
        self.model.train()
        num_proc_data = 0

        lr = self.optimizer_wrapper.get_last_lr()

        accumulated_grad = dict()
        for _ in range(self.config.NUM_LOCAL_UPDATES):
            # load mask and train model
            with torch.no_grad():
                for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                    if mask is not None:
                        layer.weight *= mask
            inputs, labels = self.train_loader.get_next_batch()
            list_grad = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            num_proc_data += len(inputs)

            # accumulate grad
            for (key, param), g in zip(self.model.named_parameters(), list_grad):
                assert param.size() == g.size()  # only simulation
                if key in accumulated_grad.keys():
                    accumulated_grad[key] += param.grad  # g
                else:
                    accumulated_grad[key] = param.grad  # g

        # accumulate square grad(epoch)
        with torch.no_grad():
            if self.use_adaptive:
                for key, grad in accumulated_grad.items():
                    if key in self.accumulated_sgrad.keys():
                        self.accumulated_sgrad[key] += (grad ** 2) * num_proc_data
                    else:
                        self.accumulated_sgrad[key] = (grad ** 2) * num_proc_data
            
            # why to load mask again?
            for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                if mask is not None:
                    layer.weight *= mask

        self.optimizer_wrapper.lr_scheduler_step()

        if self.use_adaptive and is_adj_round:
            sgrad_to_upload = deepcopy_dict(self.accumulated_sgrad)
            self.accumulated_sgrad = dict()
        else:
            sgrad_to_upload = {}
        
        # return parameters, epoch length, square grad, learning rate
        return self.model.state_dict(), num_proc_data, sgrad_to_upload, lr



