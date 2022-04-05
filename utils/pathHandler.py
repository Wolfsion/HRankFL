from abc import ABC, abstractmethod
import os
import time
import pickle
from env.preEnv import DataSetType


# 待实现优化
# 将milestone内容迁移到checkpoint对应模型下
# checkout 当目录不存在时创建
# 对象的序列化与反序列化实现

def checkout(path: str):
    if not os.path.isdir(path):
        os.makedirs(dir)


def store(path: str, _obj):
    with open(path, "wb") as f:
        pickle.dump(_obj, f)


def curt_time_stamp():
    pattern = '%Y.%m.%d_%H-%M-%S'
    return time.strftime(pattern, time.localtime(time.time()))


class PathGather(ABC):
    ERROR_MESS1 = "Given directory doesn't exists."

    def __init__(self, model, dataset, image) -> None:
        self.mpath: str = model
        self.dpath: str = dataset
        self.ipath: str = image

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def dataset(self, d_type: DataSetType):
        pass

    @abstractmethod
    def configs(self):
        pass


class HRankPathGather(PathGather):
    vgg_pt = 'vgg_16_bn.pt'

    def __init__(self, model: str, dataset: str, images: str) -> None:
        super().__init__(model, dataset, images)
        self.rank_index = 0

        self.rank_dir = 'ranks'
        self.config_dir = 'configs'
        self.visual_dir = 'visual'

    # Outer API
    def model(self, fixed=False):
        if fixed:
            return os.path.join(self.mpath, self.vgg_pt)
        else:
            model_name = curt_time_stamp() + '.pt'
            return os.path.join(self.mpath, model_name)

    def dataset(self, d_type: DataSetType):
        if d_type == DataSetType.CIFAR10:
            return os.path.join(self.dpath, "CIFAR10")
        else:
            return "null"

    def rank(self):
        self.rank_index += 1
        rank_name = str(self.rank_index) + '.npy'
        return os.path.join(self.mpath, self.rank_dir, rank_name)

    def reset_rank_index(self):
        self.rank_index = 0

    def configs(self, name: str = 'chip'):
        return os.path.join(self.mpath, self.config_dir, name)

    def img(self, name: str):
        return os.path.join(self.ipath, name)

    def visual(self, name: str):
        return os.path.join(self.visual_dir, name)







