import torch

from dl.model.modelUtil import valid_performance, vgg_16_bn
from dl.compress.trash.irank import IterRank
from env.preEnv import *
from env.runtimeEnv import *

from dl.compress.Sparse import TopKSparse
from dl.compress.Quantization import QuantizationSGD
from dl.compress.utils import get_size

# loader is not proper for single train
# global_loader = loader_pool(3, 64)
# GLOBAL_LOGGER.info("Sampler initialized----------")


def origin_model():
    ori_model = vgg_16_bn(compress_rate)
    alg_obj = IterRank(ori_model)
    alg_obj.device_train(global_loader[0], 100)

    GLOBAL_LOGGER.info('Origin model-------------')
    valid_performance(global_loader[0], alg_obj.wrapper)


def random_pruning_model():
    shrink_model = vgg_16_bn(candidate_rate)
    alg_obj = IterRank(shrink_model)
    alg_obj.device_train(global_loader[1], 100)

    GLOBAL_LOGGER.info('Random compress model-------------')
    valid_performance(global_loader[1], alg_obj.wrapper)


def hrank_pruning_model():
    ori_model = vgg_16_bn(compress_rate)
    alg_obj = IterRank(ori_model)
    alg_obj.init_cp_model(candidate_rate)
    alg_obj.get_rank(global_loader[2])
    alg_obj.load_params()

    alg_obj2 = IterRank(alg_obj.cp_model)
    alg_obj2.device_train(global_loader[2], 100)

    GLOBAL_LOGGER.info('HRank compress model-------------')
    valid_performance(global_loader[2], alg_obj2.wrapper)


def sparse_and_quan():
    tensor = torch.randn(2, 2, 2, 2)
    sparser = TopKSparse(0.5)
    quaner = QuantizationSGD(16)

    print("origin tensor:", get_size(tensor))
    ct = quaner.compress(tensor)["com_tensor"]

    print(get_size(ct))

def main():
    sparse_and_quan()
