from abc import ABC, abstractclassmethod
import os
import time

class PathGather(ABC):
    ERROR_MESS1 = "Given directory doesn't exists."
    def __init__(self, model, dataset) -> None:
        self.mpath:str = model
        self.dpath:str = dataset
        self.config_dir = 'configs'
    
    @abstractclassmethod
    def model_dir(self):
        pass

    @abstractclassmethod
    def model_name(self):
        pass

    @abstractclassmethod
    def model(self):
        pass

    @abstractclassmethod
    def dataset_dir(self):
        pass

    @abstractclassmethod
    def configs(self):
        pass

    def checkout(self, path: str):
        return os.path.isdir(path)

class HRankPathGather(PathGather):
    def __init__(self, model: str, dataset: str, ranks: str) -> None:
        super().__init__(model, dataset)
        self.rpath = ranks
        self.rank_index = 0

    def curt_time_stamp(self):
        pattern = '%Y.%m.%d_%H-%M-%S'
        return time.strftime(pattern, time.localtime(time.time()))

    def model_dir(self):
        assert self.checkout(self.mpath), self.ERROR_MESS1
        return self.mpath

    def model_name(self):
        return self.curt_time_stamp() + '.pt'

    def model(self):
        return os.path.join(self.model_dir(), self.model_name())

    def dataset_dir(self):
        assert self.checkout(self.dpath), self.ERROR_MESS1
        return self.dpath

    def rank_dir(self):
        assert self.checkout(self.rpath), self.ERROR_MESS1
        return self.rpath

    def rank_name(self):
        self.rank_index += 1
        return str(self.rank_index) + '.npy'

    def rank(self):
        return os.path.join(self.rank_dir(), self.rank_name())

    def reset_rank_index(self):
        self.rank_index = 0

    def configs(self, name: str = 'chip'):
        return os.path.join(self.model_dir(), self.config_dir, name)
