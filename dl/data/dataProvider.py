import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join

from env.preEnv import *
from env.runtimeEnv import *
from dl.data.transform import OneHot, DataToTensor


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=8,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.current_iter = self.__iter__()

    def get_next_batch(self):
        try:
            return self.current_iter.__next__()
        except StopIteration:
            self.current_iter = self.__iter__()
            return self.current_iter.__next__()

    def skip_epoch(self):
        self.current_iter = self.__iter__()

    @property
    def len_data(self):
        return len(self.dataset)


def get_data(dataset: DataSetType, data_type, transform=None, target_transform=None, user_list=None):
    if dataset == DataSetType.FEMNIST:
        pass

    elif dataset == DataSetType.CelebA:
        pass

    elif dataset == DataSetType.CIFAR10:
        assert data_type in ["train", "test"]
        if transform is None:
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                                   OneHot(CIFAR10_CLASSES, to_float=True)])

        return torchvision.datasets.CIFAR10(root=join(datasets, "CIFAR10"),
                                            train=data_type == "train", download=True,
                                            transform=transform,
                                            target_transform=target_transform)

    elif dataset == DataSetType.CIFAR100:
        assert data_type in ["train", "test"]
        if transform is None:
            mean = CIFAR100_MEAN
            std = CIFAR100_STD
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                                   OneHot(CIFAR100_CLASSES, to_float=True)])
        return torchvision.datasets.CIFAR100(root=join(datasets, "CIFAR100"),
                                             train=data_type == "train", download=True,
                                             transform=transform,
                                             target_transform=target_transform)

    elif dataset == DataSetType.ImageNet:
        assert data_type in ["train", "test"]
        imagenet_data = torchvision.datasets.ImageNet(root=join(datasets, "ImageNet"), download=True)


    else:
        raise ValueError("{} dataset is not supported.".format(dataset))


def get_data_loader(name: DataSetType, data_type: str, batch_size=None, shuffle: bool = False, sampler=None, transform=None,
                    target_transform=None, subset_indices=None, num_workers=8, pin_memory=False, user_list=None):
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        assert batch_size is not None, "Batch size for training data is required"
    if shuffle is True:
        assert sampler is None, "Cannot shuffle when using sampler"

    data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform,
                    user_list=user_list)

    if subset_indices is not None:
        data = torch.utils.data.Subset(data, subset_indices)
    if data_type != "train" and batch_size is None:
        batch_size = len(data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)
