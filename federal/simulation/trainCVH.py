import random

import data.samplers as samplers
from data.dataProvider import get_data_loader
import model.modelUtil as modelUtil
from model.vwrapper import VWrapper
from pruning.vhrank import VGG16HRank
from federal.FLnodes import *
from control.preEnv import *
from control.runtimeEnv import *

sa = samplers.CF10NIIDSampler(num_slices, MAX_ROUND, data_per_client_epoch,
                              True, client_per_round)
workers_loader = get_data_loader(CIFAR10_NAME, data_type="train",
                                 batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                 sampler=sa, num_workers=0, pin_memory=True)
GLOBAL_LOGGER.info("Sampler initialized")


# CIFAR10 VGG16 HRank
class MasterPlus(FLMaster):
    def __init__(self, args: argparse.Namespace, model: nn.Module,
                 bridge: FLSimNet = default_bridge, save_interval=50):
        self.wrapper = VWrapper(model)
        self.wrapper.model.eval()
        self.pruning_rate = None
        self.dicts = []
        self.batches = []
        super().__init__(args, bridge, save_interval)

    def save_exp_config(self):
        pass

    # ! pruning_rate hyper obtain
    def init_algorithm(self):
        self.pruning_rate = [0.45]*7 + [0.78]*5

    def distribute_dict(self, index_limit=workers - 1):
        list_mess = self.recv_mess()
        for index, mess in enumerate(list_mess):
            mess.run(dicts_container=self.dicts, batches_container=self.batches)
            if index == index_limit:
                break
        batch_sum = sum(self.batches)
        for index, batch in enumerate(self.batches):
            self.batches[index] = batch / batch_sum
        mess = FLMessage(MessType.DOWNLOAD_STATIC_DICT)
        mess.run(dicts=deepcopy(self.dicts),
                 batches=deepcopy(self.batches),
                 wrapper=self.wrapper)
        self.send_mess(mess)

    def distribute_cp_rate(self):
        mess = FLMessage(MessType.DOWNLOAD_PRUNING_RATE)
        mess.run(cp_rate=deepcopy(self.pruning_rate))
        self.send_mess(mess)


class WorkerPlus(FLWorker):
    def __init__(self, model: nn.Module):
        self.model = model
        self.alg_obj: VGG16HRank = None
        super().__init__()
        self.alg_obj.model.train()

    def init_loader(self):
        self.loader = workers_loader

    def init_algorithm(self):
        self.alg_obj = VGG16HRank(self.model)

    def push_dict(self):
        mess = FLMessage(MessType.UPLOAD_STATIC_DICT)
        mess.run(False, alg=self.alg_obj, loader=self.loader)
        self.send_mess(mess)

    def fetch_dict(self):
        mess = self.recv_mess()
        mess.run(False, alg=self.alg_obj)

    def fetch_cp_rate(self):
        mess = self.recv_mess()
        mess.run(False, alg=self.alg_obj)


# deepcopy generate model
class RunPlus:
    def __init__(self, args: argparse.Namespace) -> None:
        self.random_indices = list(range(workers))
        random.shuffle(self.random_indices)
        self.args = args
        self.pipe = FLSimNet()
        self.model = modelUtil.vgg_16_bn(ORIGIN_CP_RATE)
        self.models = [modelUtil.vgg_16_bn(ORIGIN_CP_RATE) for _ in range(workers)]

        self.master = MasterPlus(args, self.model)
        self.workers = [WorkerPlus(self.models[i]) for i in range(workers)]

    def upload_download_dict(self):
        for worker in self.workers:
            worker.push_dict()
        self.master.distribute_dict()

    def valid_acc(self):
        self.master.wrapper.device.save_model('needle.pt')
