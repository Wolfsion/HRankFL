import math

from dl.compress.vhrank import VGG16HRank
from dl.data import samplers
from dl.data.dataProvider import get_data_loader, get_data_loaders
from dl.model import modelUtil
from federal.merge.FedAvg import FedAvg

from env.preEnv import DataSetType, ORIGIN_CP_RATE, GLOBAL_LOGGER, STEP_DECAY
from env.runtimeEnv import *


def union_convergence():
    list_dict = []
    fedavg = FedAvg()
    sampler = samplers.CF10NIIDSampler(num_slices, 100, data_per_client_epoch, True, client_per_round)
    workers_loaders = get_data_loaders(DataSetType.CIFAR10, data_type="train",
                                       batch_size=batch_size, users_indices=sampler.users_indices,
                                       num_workers=0, pin_memory=True)
    test_loader = get_data_loader(DataSetType.CIFAR10, data_type="test", batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    hrank_objs = [VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE)) for _ in range(num_slices)]

    curt_train_batch = 0

    for rnd in range(100):
        GLOBAL_LOGGER.info(f"FL turn:{rnd}...")
        curt_selected = sampler.curt_selected()

        for idx in curt_selected[rnd]:
            GLOBAL_LOGGER.info(f"Train from device:{idx}")
            for i in range(local_epoch):
                curt_train_batch += hrank_objs[idx].learn_run(workers_loaders[idx])
            list_dict.append(hrank_objs[idx].interrupt_mem())

        union_dict = fedavg.merge_dict(list_dict, [1 for _ in range(client_per_round)])

        for idx in range(num_slices):
            hrank_objs[idx].restore_mem(union_dict)
            hrank_objs[idx].adjust_lr(math.pow(STEP_DECAY, curt_train_batch//10))

        hrank_objs[0].show_lr()
        GLOBAL_LOGGER.info(f"FL turn test:{rnd}...")
        hrank_objs[0].wrapper.valid_performance(test_loader)
        list_dict.clear()

    GLOBAL_LOGGER.info('Test Loader------')
    hrank_objs[0].wrapper.valid_performance(test_loader)
    hrank_objs[0].interrupt_disk('union.snap')


def main():
    union_convergence()
