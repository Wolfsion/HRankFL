from abc import ABC, abstractclassmethod
from collections import OrderedDict
from torch import Tensor
from deprecated import deprecated
import torch.nn as nn
import numpy as np

from env.preEnv import GLOBAL_LOGGER


class ModelMask(ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.init_mask()
    
    @abstractclassmethod
    def init_mask(self):
        pass

    @abstractclassmethod
    def model_density(self):
        pass
    
    @abstractclassmethod
    def update_mask(self):
        pass

# ori_state_dict: origin state dict(not contain module.)
@deprecated(version='0.1', reason="This class or function is not perfect")
class HRankMask(ModelMask): 
    def __init__(self, ori_state_dict: 'OrderedDict[str,Tensor]',
                    model: nn.Module, rank_path: str, GPUs: bool = False) -> None:
        super().__init__(model)
        self.osd = ori_state_dict
        self.rp = rank_path
        self.GPUs = GPUs
        
    def init_mask(self):
        pass

    def model_density(self):
        pass
        
    def update_mask(self):
        pass