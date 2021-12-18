from abc import ABC, abstractclassmethod
import os
import time

class PathGather(ABC):
    def __init__(self, modeldict, dataset) -> None:
        self.mpath = modeldict
        self.dpath = dataset
        self.config_dir = 'configs'
    
    @abstractclassmethod
    def modeldict_dir(self):
        pass

    @abstractclassmethod
    def modeldict_name(self):
        pass

    @abstractclassmethod
    def modeldict_dir(self):
        pass

    @abstractclassmethod
    def modeldict(self):
        pass

    @abstractclassmethod
    def dataset_dir(self):
        pass

    @abstractclassmethod
    def configs(self):
        pass

class HRankPathGather(PathGather):
    def __init__(self, modeldict, dataset, ranks) -> None:
        super().__init__(modeldict, dataset)
        self.rpath = ranks
        self.rank_index = 0

    def curt_time_stamp(self):
        pattern = '%Y.%m.%d_%H-%M-%S'
        return time.strftime(pattern, time.localtime(time.time()))

    def modeldict_dir(self):
        return self.modeldict_path

    def modeldict_name(self):
        return self.curt_time_stamp() + '.pt'

    def modeldict(self):
        return os.path.join(self.modeldict_dir(), self.modeldict_name())

    def dataset_dir(self):
        return self.dataset_path

    def rank_dir(self):
        return self.rpath

    def rank_name(self):
        self.rank_index += 1
        return str(self.rank_index) + '.npy'

    def rank(self):
        return os.path.join(self.rank_dir(), self.rank_name())

    def configs(self, name: str = 'chip'):
        return os.path.join(self.modeldict_dir(), self.config_dir, name)