import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.nn.functional import binary_cross_entropy_with_logits

from env.preEnv import *
from env.runtimeEnv import *
from dl.model import vdevice, modelUtil


# 配置保存与加载

class SGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("closure not supported")

        list_grad = []

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if (p.grad is None and not hasattr(p, "is_sparse_param")) or hasattr(p, "is_placeholder"):
                    # exclude 1) dense param with None grad and 2) dense placeholders for sparse params
                    continue
                elif hasattr(p, "is_sparse_param"):
                    d_p = p.dense.grad.masked_select(p.mask)
                    if weight_decay != 0:
                        d_p = d_p.add(p._values(), alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p._values().add_(d_p, alpha=-group['lr'])

                else:
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.add_(d_p, alpha=-group['lr'])

                list_grad.append(d_p.clone())
        return list_grad

    def clear_state(self):
        for state in self.state.values():
            if "momentum_buffer" in state:
                del state["momentum_buffer"]


class VWrapper:
    ERROR_MESS1 = "Checkpoint do not find model_key attribute"

    def __init__(self, model: nn.Module, loss=binary_cross_entropy_with_logits,
                 optim: Optimizer = None, scheduler=None,
                 device: vdevice.VADevice = None) -> None:
        self.model = model
        self.loss_func = loss
        self.device = device

        if device is None:
            self.device = vdevice.VADevice(True, gpu)
        self.model = self.device.bind_model(self.model)

        if optim is None:
            self.use_default_optim()
        else:
            self.optimizer = optim
            self.lr_scheduler = scheduler

    def use_default_optim(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                gamma=0.5 ** (1 / LR_HALF_LIFE))

    # def step(self, inputs, labels):
    #     self.zero_grad()
    #     self.model.train()
    #     inputs, labels = self.device.on_tensor(inputs, labels)
    #     pred = self.model(inputs)
    #     loss = self.loss_func(pred, labels)
    #     loss.backward()
    #     return self.optimizer.step()

    def step(self, inputs, labels, train=False):
        if train:
            self.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        test_loss = 0
        correct = 0
        inputs, labels = self.device.on_tensor(inputs, labels)
        pred = self.model(inputs)
        loss = self.loss_func(pred, labels)
        if train:
            loss.backward()
            self.optim_step()
        test_loss += loss.item()
        _, predicted = pred.max(1)
        _, targets = labels.max(1)
        correct += predicted.eq(targets).sum().item()
        return test_loss, correct

    def zero_grad(self):
        self.model.zero_grad()

    def optim_step(self):
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.lr_scheduler.get_last_lr()[0]

    def access_model(self) -> nn.Module:
        return self.device.access_model()

    # to finish
    def save_checkpoint(self, ext_info: str):
        # exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
        #                     "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
        #                     "lrhl": LR_HALF_LIFE}
        exp_const_config = {"exp_name": CIFAR10_NAME, "state_dict": self.device.freeze_model()}
        # args_config = vars(self.args)
        # configs = exp_const_config.copy()
        # configs.update(args_config)
        modelUtil.mkdir_save(exp_const_config, file_repo.configs('exp_config.snap'))

    # !
    def load_checkpoint(self, path: str, model_key: str = 'state_dict'):
        checkpoint = torch.load(path)
        assert model_key in checkpoint.keys(), self.ERROR_MESS1
        self.device.load_model(checkpoint[model_key])
