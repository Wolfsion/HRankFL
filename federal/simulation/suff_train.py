from dl.compress.vhrank import VGG16HRank, ResNet56HRank
from dl.model import modelUtil
from env.preEnv import *
from dl.data import samplers
from dl.data.dataProvider import get_data_loader
from dl.data.dataProvider import get_data
from env.runtimeEnv import *
from federal.merge.FedAvg import FedAvg


def init_datasets():
    get_data(DataSetType.CIFAR100, data_type="train")
    get_data(DataSetType.ImageNet, data_type="train")


def vgg16_cifar10_single_convergence():
    loader = get_data_loader(DataSetType.CIFAR10, data_type="train",
                             batch_size=32, shuffle=True,
                             num_workers=0, pin_memory=True)
    GLOBAL_LOGGER.info("Sampler initialized")
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    for i in range(1000):
        hrank_obj.learn_run(loader)

    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=True, num_workers=0, pin_memory=True)
    GLOBAL_LOGGER.info('Test Loader------')
    hrank_obj.wrapper.valid_performance(test_loader)

    hrank_obj.interrupt_disk('single.snap')

def resnet56_cifar100_single_convergence():
    loader = get_data_loader(DataSetType.CIFAR100, data_type="train",
                             batch_size=32, shuffle=True,
                             num_workers=0, pin_memory=True)
    GLOBAL_LOGGER.info("Sampler initialized")
    hrank_obj = ResNet56HRank(modelUtil.resnet_56(ORIGIN_CP_RATE))
    for i in range(100):
        hrank_obj.learn_run(loader)

    test_loader = get_data_loader(DataSetType.CIFAR100, data_type="test", batch_size=32,
                                  shuffle=True, num_workers=0, pin_memory=True)

    GLOBAL_LOGGER.info('Test Loader------')
    hrank_obj.wrapper.valid_performance(test_loader)
    hrank_obj.interrupt_disk('single.snap')

def union_convergence():
    list_dict = []
    union_dict = dict()
    fedavg = FedAvg()
    sampler = samplers.CF10NIIDSampler(num_slices, 100, data_per_client_epoch, True, client_per_round)
    workers_loaders = get_data_loader(CIFAR10_NAME, data_type="train",
                                      batch_size=32, shuffle=True,
                                      sampler=sampler, num_workers=0, pin_memory=True)
    hrank_objs = [VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE)) for _ in range(num_slices)]

    for rnd in range(100):
        curt_selected = sampler.curt_selected()
        for idx in curt_selected[rnd]:
            GLOBAL_LOGGER.info(f"Train from device:{idx}")
            hrank_objs[idx].learn_run(workers_loaders)
            list_dict.append(hrank_objs[idx].interrupt_mem())

        union_dict = fedavg.merge_dict(list_dict, [1 for _ in range(client_per_round)])

        for idx in range(num_slices):
            hrank_objs[idx].restore_mem(union_dict)

        list_dict.clear()

    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=True, num_workers=0, pin_memory=True)
    GLOBAL_LOGGER.info('Test Loader------')
    hrank_objs[0].wrapper.valid_performance(test_loader)
    hrank_objs[0].interrupt_disk('union.snap')


def test_checkpoint():
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    hrank_obj.restore_disk()
    train_loader = get_data_loader(CIFAR10_NAME, data_type="train", batch_size=32,
                                   shuffle=True, num_workers=4, pin_memory=True)
    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=True, num_workers=4, pin_memory=True)
    GLOBAL_LOGGER.info('Train Loader------')
    hrank_obj.wrapper.valid_performance(train_loader)
    GLOBAL_LOGGER.info('Test Loader------')
    hrank_obj.wrapper.valid_performance(test_loader)


def main():
    union_convergence()
