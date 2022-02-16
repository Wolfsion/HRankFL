from model.modelUtil import valid_performance, vgg_16_bn
from pruning.irank import IterRank
from control.preEnv import *
from control.runtimeEnv import *
import data.samplers as samplers
from data.dataProvider import get_data_loader

# loader is not proper for single train
sa = samplers.CF10NIIDSampler(3, MAX_ROUND, data_per_client_epoch, True, 3)
global_loader = get_data_loader(CIFAR10_NAME, data_type="train",
                                batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                sampler=sa, num_workers=8, pin_memory=False)

GLOBAL_LOGGER.info("Sampler initialized----------")


def origin_model():
    GLOBAL_LOGGER.info('Origin model-------------')
    ori_model = vgg_16_bn(compress_rate)
    alg_obj = IterRank(ori_model)
    alg_obj.device_train(global_loader, 100)
    valid_performance(alg_obj.model)


def random_pruning_model():
    GLOBAL_LOGGER.info('Random pruning model-------------')
    shrink_model = vgg_16_bn(candidate_rate)
    alg_obj = IterRank(shrink_model)
    alg_obj.device_train(global_loader, 100)
    valid_performance(alg_obj.model)


def hrank_pruning_model():
    GLOBAL_LOGGER.info('HRank pruning model-------------')
    ori_model = vgg_16_bn(compress_rate)
    alg_obj = IterRank(ori_model)
    alg_obj.init_cp_model()
    alg_obj.get_rank(global_loader)
    alg_obj.load_params()

    alg_obj2 = IterRank(alg_obj.cp_model)
    alg_obj2.device_train(global_loader, 100)
    valid_performance(alg_obj2.model)

def main():
    origin_model()
    random_pruning_model()
    hrank_pruning_model()
