from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from visual.DataExtractor import Extractor
from env.runtimeEnv import *

# axis mode:
'''
    F:FLOPs
    A:Acc
    T:Turn
    R:Rate
'''

# form mode:
'''
    h:histogram-4
    k:kernel density estimation-2
    r:rug-1
'''


class VisBoard:
    ERROR_MESS1 = "The form length cannot exceed 3 characters."
    ERROR_MESS2 = "The mode str contains not defined character."
    map_dict = {"h": 4, "k": 2, "r": 1}

    def __init__(self, data_resource: Extractor):
        self.io = data_resource

    def map_int(self, form: str) -> int:
        assert len(form) < 4, self.ERROR_MESS1
        assert all([ch in self.map_dict.keys() for ch in form]), self.ERROR_MESS2
        ret = 0
        for ch in form:
            ret += self.map_dict[ch]
        return ret

    def single_var_dist(self, axis: str, form: str):
        mode = self.map_int(form)
        data_list = self.io.map_vars(axis)
        x = [1, 2, 3, 4]
        sns.displot(x)
        plt.savefig(file_repo.img("test"))

    def double_vars_dist(self, axis: str, form: str):
        mode = self.map_int(form)

    def double_vars_regression(self, axis: str, form: str):
        mode = self.map_int(form)

    def multi_vars_regression(self, axis: str, form: str):
        mode = self.map_int(form)
