from abc import ABC, abstractmethod
import os
import time


def checkout(path: str):
    return os.path.isdir(path)


class PathGather(ABC):
    ERROR_MESS1 = "Given directory doesn't exists."

    def __init__(self, model, dataset) -> None:
        self.mpath: str = model
        self.dpath: str = dataset
        self.config_dir = 'configs'
        self.visual_dir = 'visual'

    @abstractmethod
    def model_dir(self):
        pass

    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def dataset_dir(self):
        pass

    @abstractmethod
    def configs(self):
        pass


def curt_time_stamp():
    pattern = '%Y.%m.%d_%H-%M-%S'
    return time.strftime(pattern, time.localtime(time.time()))


class HRankPathGather(PathGather):
    vgg_pt = 'vgg_16_bn.pt'

    def __init__(self, model: str, dataset: str, ranks: str, images: str) -> None:
        super().__init__(model, dataset)
        self.rpath = ranks
        self.ipath = images
        self.rank_index = 0

    def model_dir(self):
        assert checkout(self.mpath), self.ERROR_MESS1
        return self.mpath

    def model_name(self):
        return curt_time_stamp() + '.pt'

    def model(self, fixed=False):
        if fixed:
            return os.path.join(self.model_dir(), self.vgg_pt)
        else:
            return os.path.join(self.model_dir(), self.model_name())

    def dataset_dir(self):
        assert checkout(self.dpath), self.ERROR_MESS1
        return self.dpath

    def rank_dir(self):
        assert checkout(self.rpath), self.ERROR_MESS1
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

    def img_dir(self):
        assert checkout(self.ipath), self.ERROR_MESS1
        return self.ipath

    def img(self, name: str):
        return os.path.join(self.img_dir(), name)

    def visual(self, name: str):
        return os.path.join(self.model_dir(), self.config_dir, name)
