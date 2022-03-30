# init loader
# init vhrank
# run federal learning
# save
from dl.compress.vhrank import VGG16HRank
from dl.model import modelUtil
from env.preEnv import *
from dl.data import samplers
from dl.data.dataProvider import get_data_loader


def single_convergence():
    sampler = samplers.CF10NIIDSampler(1, 10001, 32, True, 1)
    loader = get_data_loader(CIFAR10_NAME, data_type="train",
                             batch_size=32, shuffle=False,
                             sampler=sampler, num_workers=0, pin_memory=True)
    GLOBAL_LOGGER.info("Sampler initialized")
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    hrank_obj.learn_run(loader)

def union_convergence():
    pass


def main():
    single_convergence()
