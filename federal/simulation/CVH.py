import random

from dl import data as samplers
from dl.data import get_data_loader
import dl.model.modelUtil as modelUtil
from dl.model.vwrapper import VWrapper
from dl.compress import VGG16HRank
from federal.FLnodes import *
from env.preEnv import *
from env.runtimeEnv import *

sa = samplers.CF10NIIDSampler(num_slices, MAX_ROUND, data_per_client_epoch,
                              True, client_per_round)
workers_loader = get_data_loader(CIFAR10_NAME, data_type="train",
                                 batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                 sampler=sa, num_workers=0, pin_memory=True)
GLOBAL_LOGGER.info("Sampler initialized")


# CIFAR10 VGG16 HRank
class CVHMaster(FLMaster):
    def __init__(self, args: argparse.Namespace, model: nn.Module,
                 bridge: FLSimNet = default_bridge, save_interval=50):
        self.wrapper = VWrapper(model)
        self.wrapper.model.eval()
        self.ranks = []
        self.pruning_rate = None
        super().__init__(args, bridge, save_interval)

    # ! pruning_rate hyper obtain
    def init_algorithm(self):
        self.wrapper.load_checkpoint(file_repo.model(fixed=True))
        self.pruning_rate = [0.45]*7 + [0.78]*5

    # ! replace by wrapper.method()
    def save_exp_config(self):
        exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
                            "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
                            "lrhl": LR_HALF_LIFE}
        args_config = vars(self.args)
        configs = exp_const_config.copy()
        configs.update(args_config)
        modelUtil.mkdir_save(configs, file_repo.configs('exp_config.snap'))

    def distribute_dict(self):
        mess = FLMessage(MessType.DOWNLOAD_STATIC_DICT)
        mess.run(model=deepcopy(self.wrapper.model))
        self.send_mess(mess)

    def distribute_rank(self, index_limit=workers-1):
        list_mess = self.recv_mess()
        for index, mess in enumerate(list_mess):
            mess.run(ranks_container=self.ranks)
            if index == index_limit:
                break
        mess = FLMessage(MessType.DOWNLOAD_RANK)
        mess.run(ranks=deepcopy(self.ranks))
        self.send_mess(mess)

    def distribute_cp_rate(self):
        mess = FLMessage(MessType.DOWNLOAD_PRUNING_RATE)
        mess.run(cp_rate=deepcopy(self.pruning_rate))
        self.send_mess(mess)


class CVHWorker(FLWorker):
    def __init__(self, model: nn.Module):
        self.model = model
        self.alg_obj: VGG16HRank = None
        super().__init__()
        self.alg_obj.model.train()
        self.rank = None

    def init_loader(self):
        self.loader = workers_loader

    def init_algorithm(self):
        self.alg_obj = VGG16HRank(self.model)

    def fetch_dict(self):
        mess = self.recv_mess()
        mess.run(False, alg=self.alg_obj)

    def fetch_cp_rate(self):
        mess = self.recv_mess()
        mess.run(False, alg=self.alg_obj)

    def fetch_rank(self):
        mess = self.recv_mess()
        mess.run(False, alg=self.alg_obj)

    def push_rank(self):
        mess = FLMessage(MessType.UPLOAD_RANK)
        mess.run(False, alg=self.alg_obj, loader=self.loader)
        self.send_mess(mess)


# deepcopy generate model
class CVHRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.random_indices = list(range(workers))
        random.shuffle(self.random_indices)
        self.args = args
        self.pipe = FLSimNet()
        self.model = modelUtil.vgg_16_bn(ORIGIN_CP_RATE)
        self.models = [modelUtil.vgg_16_bn(ORIGIN_CP_RATE) for _ in range(workers)]
        self.master = CVHMaster(args, self.model)
        self.workers = [CVHWorker(self.models[i]) for i in range(workers)]

    def download_dict(self):
        self.master.distribute_dict()
        for worker in self.workers:
            worker.fetch_dict()

    def download_cp_rate(self):
        self.master.distribute_cp_rate()
        for worker in self.workers:
            worker.fetch_cp_rate()

    def upload_download_ranks(self):
        for worker in self.workers:
            worker.push_rank()
        if self.args.cs:
            self.master.distribute_rank(self.args.cs)
        else:
            self.master.distribute_rank()
        for worker in self.workers:
            worker.fetch_rank()

    def valid_acc(self):
        self.workers[0].alg_obj.wrapper.device.load_model('./test.pt')
