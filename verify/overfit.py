import math

from dl.compress.vhrank import VGG16HRank
from dl.data import samplers
from dl.data.dataProvider import get_data_loader
from dl.model import modelUtil
from federal.merge.FedAvg import FedAvg

from env.preEnv import DataSetType, ORIGIN_CP_RATE, GLOBAL_LOGGER, STEP_DECAY
from env.runtimeEnv import *


def union_convergence():
    list_dict = []
    fedavg = FedAvg()
    sampler = samplers.CF10NIIDSampler(num_slices, 100, data_per_client_epoch, True, client_per_round)

    workers_loader = get_data_loader(DataSetType.CIFAR10, data_type="train",
                                     batch_size=batch_size, shuffle=False,
                                     sampler=sampler, num_workers=0, pin_memory=True)
    test_loader = get_data_loader(DataSetType.CIFAR10, data_type="test", batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    hrank_objs = [VGG16HRank(modelUtil.vgg_16_bn(ORIGIN_CP_RATE)) for _ in range(num_slices)]

    for rnd in range(10):
        GLOBAL_LOGGER.info(f"FL turn:{rnd}...")
        curt_selected = sampler.curt_selected()

        for idx in curt_selected[rnd]:
            GLOBAL_LOGGER.info(f"Train from device:{idx}")
            hrank_objs[idx].learn_run(workers_loader)
            list_dict.append(hrank_objs[idx].interrupt_mem())

        union_dict = fedavg.merge_dict(list_dict, [1 for _ in range(client_per_round)])

        for idx in range(num_slices):
            hrank_objs[idx].restore_mem(union_dict)
            hrank_objs[idx].adjust_lr(math.pow(STEP_DECAY, (client_per_round - 1) * union_train_limit))

        list_dict.clear()

    GLOBAL_LOGGER.info('Test Loader------')
    hrank_objs[0].wrapper.valid_performance(test_loader)
    hrank_objs[0].interrupt_disk('union.snap')


def main():
    union_convergence()
