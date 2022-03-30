# init loader
# init vhrank
# run federal learning
# save
from dl.compress.vhrank import VGG16HRank
from dl.model import modelUtil
from env.preEnv import *
from dl.data import samplers
from dl.data.dataProvider import get_data_loader
from dl.data.dataProvider import get_data

def init_datasets():
    get_data(DataSetType.CIFAR100, data_type="train")
    get_data(DataSetType.ImageNet, data_type="train")

def single_convergence():
    loader = get_data_loader(CIFAR10_NAME, data_type="train",
                             batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)
    GLOBAL_LOGGER.info("Sampler initialized")
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    hrank_obj.learn_run(loader)

def union_convergence():
    sampler = samplers.CF10NIIDSampler(100, 10001, 32, True, 10)
    pass


def main():
    single_convergence()
