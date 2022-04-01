from dl.compress.vhrank import VGG16HRank
from dl.model import modelUtil
from env.preEnv import *
from dl.data import samplers
from dl.data.dataProvider import get_data_loader
from dl.data.dataProvider import get_data
from env.runtimeEnv import *
from federal.merge.FedAvg import FedAvg
import dictdiffer


def init_datasets():
    get_data(DataSetType.CIFAR100, data_type="train")
    get_data(DataSetType.ImageNet, data_type="train")


def single_convergence():
    loader = get_data_loader(CIFAR10_NAME, data_type="train",
                             batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)
    GLOBAL_LOGGER.info("Sampler initialized")
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    for i in range(5):
        hrank_obj.learn_run(loader)


def union_convergence():
    list_dict = []
    union_dict = dict()
    fedavg = FedAvg()
    sampler = samplers.CF10NIIDSampler(num_slices, 100, data_per_client_epoch, True, client_per_round)
    workers_loaders = get_data_loader(CIFAR10_NAME, data_type="train",
                                      batch_size=32, shuffle=False,
                                      sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=False, num_workers=4, pin_memory=True)

    hrank_objs = [VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE)) for _ in range(num_slices)]

    for rnd in range(10):
        curt_selected = sampler.curt_selected()
        for idx in curt_selected[rnd]:
            hrank_objs[idx].learn_run(workers_loaders)
            list_dict.append(hrank_objs[idx].interrupt_mem())

        union_dict = fedavg.merge_dict(list_dict, [1 for _ in range(client_per_round)])

        for idx in range(num_slices):
            hrank_objs[idx].restore_mem(union_dict)

        list_dict.clear()

    train_loader = get_data_loader(CIFAR10_NAME, data_type="train", batch_size=32,
                                   shuffle=False, num_workers=4, pin_memory=True)
    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=False, num_workers=4, pin_memory=True)
    GLOBAL_LOGGER.info('Train Loader------')
    hrank_objs[0].wrapper.valid_performance(train_loader)
    GLOBAL_LOGGER.info('Test Loader------')
    hrank_objs[0].wrapper.valid_performance(test_loader)

    dict1 = hrank_objs[0].wrapper.model.state_dict()
    dict2 = union_dict
    GLOBAL_LOGGER('compare dict......')
    for diff in list(dictdiffer.diff(dict1, dict2)):
        print(diff)



def test_checkpoint():
    hrank_obj = VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE))
    hrank_obj.restore_disk()
    train_loader = get_data_loader(CIFAR10_NAME, data_type="train", batch_size=32,
                                     shuffle=False, num_workers=4, pin_memory=True)
    test_loader = get_data_loader(CIFAR10_NAME, data_type="test", batch_size=32,
                                  shuffle=False, num_workers=4, pin_memory=True)
    GLOBAL_LOGGER.info('Train Loader------')
    hrank_obj.wrapper.valid_performance(train_loader)
    GLOBAL_LOGGER.info('Test Loader------')
    hrank_obj.wrapper.valid_performance(test_loader)


def main():
    union_convergence()
