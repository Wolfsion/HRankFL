import numpy as np
from env.runtimeEnv import *


class Extractor:
    ERROR_MESS1 = "The length cannot exceed 4 characters."
    ERROR_MESS2 = "The mode str contains not defined character."

    def __init__(self):
        self.infor_vars = {"F": [], "A": [], "T": [], "R": []}

    def parse(self):
        pass

    def map_vars(self, axis: str) -> np.ndarray:
        assert len(axis) < 5, self.ERROR_MESS1
        assert all([ch in self.infor_vars.keys() for ch in axis]), self.ERROR_MESS2
        return np.ndarray([0])
