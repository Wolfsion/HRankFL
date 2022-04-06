import re
import random

import numpy as np
from typing import List

from utils.VContainer import VContainer
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
    log_path = 'logs/hrankFL.log'
    KEYS = ['F', 'A', 'I', 'R']
    REG_PATTERNS = [r'(?<=#FLOPs:)[^\#]+(?=#)',
                    r'(?<=#Acc:)[^\#]+(?=#)',
                    r'(?<=#Interval:)[^\#]+(?=#)',
                    r'(?<=#Rate:)[^\#]+(?=#)']
    FLOPs = 0
    Acc = 1
    Interval = 2
    Rate = 3

    def __init__(self):
        # self.info_vars = {"F": [], "A": [], "T": [], "R": []}
        self.info_vars = VContainer()

    def map_list(self, mode: str) -> List[int]:
        indices = []
        for ch in mode:
            indices.append(self.KEYS.index(ch))
        return indices

    def parse(self, indices: list):
        with open(self.log_path, "rb") as f:
            for line in f:
                for idx in indices:
                    matches = re.finditer(self.REG_PATTERNS[idx], str(line), re.M)
                    for ma in matches:
                        self.info_vars.flash(self.KEYS[idx], float(ma.group()))
                        self.info_vars.flash('class', 'VGG16')

    def map_vars(self, axis: str) -> dict:
        assert len(axis) < 5, self.ERROR_MESS1
        assert all([ch in self.KEYS for ch in axis]), self.ERROR_MESS2
        indices = self.map_list(axis)
        self.parse(indices)
        for idx in indices:
            self.info_vars.store(self.KEYS[idx])
        return self.info_vars.container
