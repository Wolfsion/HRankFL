from collections import OrderedDict
from env.runtimeEnv import HRankPathGather

class VContainer:
    def __init__(self):
        self.container = OrderedDict()

    def flash(self, key:str):

    def store(self, key:str):
        path = HRankPathGather.configs()
