from dl.data.samplers import CF10NormSamplerPool
from dl.data.dataProvider import get_data_loader, DataLoader
from env.preEnv import CIFAR10_NAME


def loader_pool(num_slices: int, batch_size: int) -> [DataLoader]:
    sampler_pool = CF10NormSamplerPool(num_slices)
    loader_list = [get_data_loader(CIFAR10_NAME, data_type="train",
                                   batch_size=batch_size, shuffle=False,
                                   sampler=sampler_pool.get_sampler(i),
                                   num_workers=8, pin_memory=False) for i in range(num_slices)]
    return loader_list
