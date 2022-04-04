import random

import numpy as np
from env.runtimeEnv import *
species = ['VGG', 'ResNet']

def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list

def random_label(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.choice(species))
    return random_int_list

def get_ori_lists():
    return [range(100), random_list(), random_list(), random_label()]

def get_ori_dict():
    return {'index':np.array(range(100)),
            'acc':np.array(random_list()),
            'rate':np.array(random_list()),
            'class':np.array(random_label())}

def get_lists():
    return list(zip(range(100), random_list(), random_list(), random_label()))


class Extractor:
    ERROR_MESS1 = "The length cannot exceed 4 characters."
    ERROR_MESS2 = "The mode str contains not defined character."

    def __init__(self):
        self.infor_vars = {"F": [], "A": [], "T": [], "R": []}

    def parse(self):
        pass

    def map_vars(self, axis: str) -> dict:
        assert len(axis) < 5, self.ERROR_MESS1
        assert all([ch in self.infor_vars.keys() for ch in axis]), self.ERROR_MESS2
        # test for unit
        test_dict = get_ori_dict()
        # test for unit
        return test_dict
