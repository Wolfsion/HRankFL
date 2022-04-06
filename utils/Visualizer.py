from matplotlib import pyplot as plt
import seaborn as sns

from utils.DataExtractor import Extractor
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
    KEYS = ['F', 'A', 'I', 'R']

    def __init__(self, data_resource: Extractor):
        self.io = data_resource
        sns.set_style('darkgrid')

    def map_int(self, form: str) -> tuple:
        assert len(form) < 4, self.ERROR_MESS1
        assert all([ch in self.map_dict.keys() for ch in form]), self.ERROR_MESS2
        ret = 0
        for ch in form:
            ret += self.map_dict[ch]
        hist_flag = False if ret < 4 else True
        kde_flag = False if ret//2 % 2 == 0 else True
        rug_flag = False if ret % 2 == 0 else True
        return hist_flag, kde_flag, rug_flag

    def text_info(self):
        plt.title('Reg')
        plt.xlabel('acc')
        plt.ylabel('rate')

    def single_var_dist(self, axis: str, form: str):
        mode = self.map_int(form)
        df = self.io.map_vars(axis)
        if mode[0]:
            sns.displot(df, x="acc", kde=mode[1], rug=mode[2], hue="class")
        else:
            sns.displot(df, x="acc", kind="kde", hue="class")
        plt.savefig(file_repo.img("test"))

    def double_vars_dist(self, axis: str, form: str):
        mode = self.map_int(form)

        df = self.io.map_vars(axis)

        sns.jointplot(data=df, x='acc', y='rate', hue="class")

        plt.savefig(file_repo.img("test"))

    def double_vars_regression(self, axis: str, form: str):
        mode = self.map_int(form)

        df = self.io.map_vars(axis)

        sns.regplot(data=df, x='acc', y='rate')
        self.text_info()
        plt.savefig(file_repo.img("test"))

    def multi_vars_regression(self, axis: str, form: str):
        mode = self.map_int(form)

        sns.lmplot()
