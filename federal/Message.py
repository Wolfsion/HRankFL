import torch.nn as nn
from copy import deepcopy

from control.preEnv import *

class FLMessage():
    ERROR_MESS1 = "Message is not legal."
    ERROR_MESS2 = "Invalid function argument."

    def __init__(self, message_type: Enum) -> None:
        assert self.legal(message_type), self.ERROR_MESS1
        self.mess = message_type
        self.func = None
        self.content = {}
        
    def legal(self, mess: Enum):
        if (abs(mess.value) > MessType.LOWWER.value and 
            abs(mess.value) < MessType.UPPER.value):
            return True
        else:
            return False

    def run(self, *args, **kwargs):
        if self.func is None:
            name_base = "op" if self.mess.value > 0 else "op_"
            fun_name = name_base + self.mess.value
            self.func = getattr(self, fun_name, self.op_default)
        self.func(*args, **kwargs)

    def op1(self, *args, **kwargs):
        assert "model" in kwargs.keys(), self.ERROR_MESS2
        model:nn.Module = kwargs["model"]
        self.content["static_dict"] = deepcopy(model.load_state_dict())

    def op_1(self, *args, **kwargs):
        assert "model" in kwargs.keys(), self.ERROR_MESS2
        model:nn.Module = kwargs["model"]
        self.content["static_dict"] = deepcopy(model.load_state_dict())
        
    def op2(self, *args, **kwargs):
        pass
    
    def op_2(self, *args, **kwargs):
        pass

    def op3(self, *args, **kwargs):
        pass

    def op_3(self, *args, **kwargs):
        pass

    def op_default(self):
        pass
    