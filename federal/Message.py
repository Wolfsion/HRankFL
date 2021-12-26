import numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

from control.preEnv import *


# Message type and Message operation
def legal(mess: Enum):
    if MessType.LOWWER.value < abs(mess.value) < MessType.UPPER.value:
        return True
    else:
        return False


class FLMessage:
    ERROR_MESS1 = "Message is not legal."
    ERROR_MESS2 = "Invalid function argument."
    ERROR_MESS3 = "Invalid message content."

    def __init__(self, message_type: Enum) -> None:
        assert legal(message_type), self.ERROR_MESS1
        self.mess = message_type
        self.func = None
        self.content = {}

    def run(self, master=True, *args, **kwargs):
        if self.func is None:
            name_base = "op" if self.mess.value > 0 else "op_"
            fun_name = name_base + self.mess.value
            self.func = getattr(self, fun_name, self.op_default)
        self.func(master, *args, **kwargs)

    def op1(self, master=True, *args, **kwargs):
        if master:
            pass
        else:
            pass

    def op_1(self, master=True, *args, **kwargs):
        if master:
            assert 'model' in kwargs.keys(), self.ERROR_MESS2
            model: nn.Module = kwargs['model']
            self.content['state_dict'] = model.state_dict()
        else:
            assert 'state_dict' in self.content.keys(), self.ERROR_MESS3
            assert 'alg' in kwargs.keys(), self.ERROR_MESS2
            kwargs['alg'].model.load_state_dict(self.content['state_dict'])

    def op2(self, master=True, *args, **kwargs):
        if master:
            assert 'ranks' in self.content.keys(), self.ERROR_MESS3
            assert 'ranks_container' in kwargs.keys(), self.ERROR_MESS2
            kwargs['ranks_container'].append(self.content['ranks'])
        else:
            assert 'alg' in kwargs.keys(), self.ERROR_MESS2
            self.content['ranks'] = deepcopy(kwargs['alg'].rank_dict)

    def op_2(self, master=True, *args, **kwargs):
        if master:
            assert 'ranks' in kwargs.keys(), self.ERROR_MESS2
            merge_rank = OrderedDict()
            for rank in kwargs['ranks']:
                for k, v in rank:
                    if k in merge_rank.keys():
                        merge_rank[k] += v
                    else:
                        merge_rank[k] = v
            self.content['rank'] = deepcopy(merge_rank)
        else:
            assert 'rank' in self.content.keys(), self.ERROR_MESS3
            assert 'alg' in kwargs.keys(), self.ERROR_MESS2
            kwargs['alg'].load_params(self.content['rank'])

    def op3(self, master=True, *args, **kwargs):
        pass

    def op_3(self, master=True, *args, **kwargs):
        if master:
            assert 'cp_rate' in kwargs.keys(), self.ERROR_MESS2
            self.content['cp_rate'] = kwargs['cp_rate']
        else:
            assert 'cp_rate' in self.content.keys(), self.ERROR_MESS3
            assert 'alg' in kwargs.keys(), self.ERROR_MESS2
            kwargs['alg'].init_cp_model(self.content['cp_rate'])

    def op_default(self):
        pass
