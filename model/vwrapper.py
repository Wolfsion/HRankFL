import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.nn.functional import binary_cross_entropy_with_logits

from control.preEnv import *
from model import vdevice

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


class VWrapper():
    def __init__(self, model: nn.Module, loss = binary_cross_entropy_with_logits,
                    optim: Optimizer = None, scheduler = None, 
                    device:vdevice.VADevice = None) -> None:
        self.model = model
        self.loss_func = loss
        self.device = device
        if device == None:
            self.device = vdevice.VADevice(True, [0])
        self.device.bind_model(self.model)
        
        if optim == None:
            self.use_default_optim()
        else:
            self.optimizer = optim
            self.lr_scheduler = scheduler

    def use_default_optim(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.lr_scheduler= lr_scheduler.StepLR(self.optimizer, step_size=1, 
                                                gamma=0.5 ** (1 / LR_HALF_LIFE))

    def step(self, inputs, labels):
        self.zero_grad()
        self.model.train()
        self.device.on_tensor(inputs, labels)
        pred = self.model(inputs)
        loss = self.loss_func(pred, labels)
        loss.backward()
        return self.optimizer.step()

    def step_eva(self, inputs, labels):
        self.model.eval()
        test_loss = 0
        correct = 0
        inputs, labels = self.device.on_tensor(inputs, labels)
        pred = self.model(inputs)
        loss = self.loss_func(pred, labels)
        test_loss = 0
        test_loss += loss.item()
        _, predicted = pred.max(1)
        correct += predicted.eq(labels).sum().item()
        return test_loss, correct

    def zero_grad(self):
        self.model.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.lr_scheduler.get_last_lr()[0]

