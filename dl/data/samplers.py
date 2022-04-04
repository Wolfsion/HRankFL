import random
from typing import List

import torch
from abc import ABC, abstractmethod
from torch.utils.data import Sampler
from fedlab.utils.dataset.partition import CIFAR10Partitioner

from env.preEnv import *
from dl.data.dataProvider import get_data


class NSampler(Sampler):
    ERROR_MESS1 = "The dataset is not supported."
    ERROR_MESS2 = "The index is out of bound."

    def __init__(self, dataset: DataSetType, indices: []):
        self.dataset = dataset
        self.is_legal()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def is_legal(self):
        assert self.dataset.value > DataSetType.LOWWER.value, self.ERROR_MESS1
        assert self.dataset.value < DataSetType.UPPER.value, self.ERROR_MESS1


class CF10NormSamplerPool:
    def __init__(self, num_slices: int, seed: int = 1):
        self.clients = num_slices
        cifar10 = get_data(DataSetType.CIFAR10.name, data_type="train")
        balance_iid_part = CIFAR10Partitioner(cifar10.targets, num_slices, balance=True,
                                              partition="iid", seed=seed)
        tmp_dict = balance_iid_part.client_dict
        self.samplers = [NSampler(DataSetType.CIFAR10, tmp_dict[i]) for i in range(num_slices)]

    def get_sampler(self, index: int) -> NSampler:
        assert index < self.clients, self.ERROR_MESS2
        return self.samplers[index]


class LSampler(Sampler, ABC):
    ERROR_MESS1 = "The dataset is not supported."

    def __init__(self, datatype: DataSetType, num_slices, num_round, data_per_client,
                 client_selection, client_per_round=None):
        self.indices = []
        self.getIndices(datatype, num_slices, num_round, data_per_client,
                        client_selection, client_per_round)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    @abstractmethod
    def getIndices(self, datatype, num_slices, num_round, data_per_client,
                   client_selection, client_per_round):
        pass


# create indices list, reorder data
class IIDSampler(LSampler):
    def getIndices(self, datatype, num_slices, num_round, data_per_client, client_selection, client_per_round):
        if datatype == DataSetType.CIFAR10.value:
            total_num = CIFAR10_NUM_TRAIN_DATA
        else:
            total_num = 0

        rand_perm = torch.randperm(total_num).tolist()
        len_slice = total_num // num_slices
        tmp_indices = []

        for i in range(num_slices):
            tmp_indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        range_partition = list(range(num_slices))
        new_list_ind = [[] for _ in range(num_slices)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_slices

        list_pos = [0] * num_slices
        for _ in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition

        for client_idx in selected_client_idx:
            ind = tmp_indices[client_idx]
            pos = list_pos[client_idx]
            while len(new_list_ind[client_idx]) < pos + data_per_client:
                random.shuffle(ind)
                new_list_ind[client_idx].extend(ind)
            self.indices.extend(new_list_ind[client_idx][pos:pos + data_per_client])
            list_pos[client_idx] = pos + data_per_client


class CF10NIIDSampler(LSampler):
    ERROR_MESS1 = "The idx_selected is null."

    def __init__(self, num_slices, num_round, data_per_client, client_selection: bool,
                 client_per_round=None, seed=1, datatype=DataSetType.CIFAR10):
        self.seed = seed
        self.idx_selected = []
        super().__init__(datatype, num_slices, num_round, data_per_client,
                         client_selection, client_per_round)

    def getIndices(self, datatype, num_slices, num_round, data_per_client, client_selection, client_per_round):
        assert datatype == DataSetType.CIFAR10, "must be CIFAR10"
        cifar10 = get_data(DataSetType.CIFAR10, data_type="train")
        hetero_dir_part = CIFAR10Partitioner(cifar10.targets, num_slices,
                                             balance=None, partition="dirichlet",
                                             dir_alpha=0.3, seed=self.seed)
        tmp_indices = hetero_dir_part.client_dict
        range_partition = list(range(num_slices))
        new_list_ind = [[] for _ in range(num_slices)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_slices

        list_pos = [0] * num_slices
        for _ in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
                self.idx_selected.append(selected_client_idx)
            else:
                selected_client_idx = range_partition

            for client_idx in selected_client_idx:
                ind = tmp_indices[client_idx]
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + data_per_client:
                    new_list_ind[client_idx].extend(ind)
                self.indices.extend(new_list_ind[client_idx][pos:pos + data_per_client])
                list_pos[client_idx] = pos + data_per_client

    def curt_selected(self) -> List[List[int]]:
        assert len(self.idx_selected) != 0, self.ERROR_MESS1
        return self.idx_selected
