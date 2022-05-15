import torch
import torch.nn
import numpy as np
from pyhessian import hessian
from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import comb
from env.preEnv import GLOBAL_LOGGER
from dl.compress.StatAnalyzer import Analyzer, _get_ratio
from torch.utils.data import DataLoader, Dataset
from dl.model.vwrapper import VWrapper


class IntervalProvider:
    SIMP_LEN = 2
    NULL_NUM = -11111
    ERROR_MESS1 = "Call is_timing_simple, the len of ranks_dict list must be 2."

    def __init__(self, tiny_const: float = 0.15, sum_limit: float = 0.1):
        self.tiny_limit = tiny_const
        self.sum_limit = sum_limit
        self.cont_list = []

    def is_timing(self, clients_ranks_dict: list, num_clients: int = None):
        if num_clients is None:
            num_clients = len(clients_ranks_dict)
        num_conv_layers = len(clients_ranks_dict[0].keys())
        combine_num = comb(num_clients, 2)
        clients_ranks = [[] for _ in range(num_conv_layers)]
        for i in range(num_clients):
            for j in range(num_conv_layers):
                clients_ranks[j].append(clients_ranks_dict[i][j])

        sum_distance = 0
        for j in range(num_conv_layers):
            cos_distance = pairwise_distances(clients_ranks[j], metric="cosine")
            for row in range(num_clients):
                for col in range(row + 1, num_clients):
                    if cos_distance[row][col] > self.tiny_limit:
                        return False
                    sum_distance += cos_distance[row][col]

        GLOBAL_LOGGER.info(f"#Interval:{sum_distance}#")
        if sum_distance > combine_num * num_conv_layers * self.sum_limit:
            return False
        else:
            return True

    def is_timing_simple(self, ranks_dict: list = None):
        if ranks_dict is None:
            ranks_dict = self.cont_list
        if len(ranks_dict) == self.SIMP_LEN:
            return self.is_timing(ranks_dict, self.SIMP_LEN)
        else:
            GLOBAL_LOGGER.info(f"#len of cont_list is not 2.#")
            return self.NULL_NUM

    def push_simp_container(self, ranks: dict):
        if len(self.cont_list) < 2:
            self.cont_list.append(ranks)
        else:
            self.cont_list[0] = self.cont_list[1]
            self.cont_list[1] = ranks


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)



def is_conv(layer):
    # DenseConv2d代替torch.nn.Conv2d
    return isinstance(layer, torch.nn.Conv2d)


class RateProvider:
    def __init__(self, wrapper: VWrapper, analyzer: Analyzer, clients: int = 100):
        self.wrapper = wrapper
        self.analyzer = analyzer
        self.global_prune_rate = 0
        self.clients = clients
        self.rates = self.analyzer.total_rates()
        self.wrapper.get_prunable_layers()

    def get_rate_for_each_layers(self):
        """
        根据模型权重的范式以及全局剪枝率，为每一层卷积层确定剪枝率
        :param model:
        :param global_prune_rate:
        :return:
        """
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in self.wrapper.model.named_parameters()
                   if v.requires_grad}
        # flat the weights
        weight_flat = np.concatenate([v.flatten() for k, v in weights.items()])
        # get the thredsheld
        number_of_weights_to_prune = int(np.ceil(self.global_prune_rate * weight_flat.shape[0]))
        threshold = np.sort(np.abs(weight_flat))[number_of_weights_to_prune]
        compress_rate = []

        # model: self.wrapper.model.prunable_layers
        # layer(DenseLayer):
        for i, layer in enumerate(self.wrapper.prunable_layers):
            if is_conv(layer):
                a = np.abs(layer.weight.data.clone().cpu().detach().numpy()) < threshold
                b = int(a.sum())
                c = layer.weight.data.numel()
                compress_rate.append(round(b / c, 4))

        return compress_rate

    def get_ratio(self, idxs):
        """
        根据特征值的概率密度，获取剪枝率
        :param idxs:
        :return:
        """
        print("=== 获取剪枝率 ===")
        self.wrapper.model.eval()
        train_loader = DataLoader(DatasetSplit(self.analyzer.train_dataset, idxs), batch_size=128, shuffle=True)
        inputs, targets = next(iter(train_loader))
        inputs, labels = self.wrapper.device.on_tensor(inputs, targets)
        hessian_comp = hessian(self.wrapper.model, self.wrapper.loss_func,
                               data=(inputs, targets), cuda=self.wrapper.device.last_choice)
        density_eigen, density_weight = hessian_comp.density(iter=50, n_v=1)
        inc = 0.1
        while True:
            t = 0.000
            ratios = []
            flag = False
            while True:
                ratio = _get_ratio(t, density_eigen=density_eigen, density_weight=density_weight)
                ratios.append(ratio)
                if ratio < ratios[0] / 2:
                    break
                if len(ratios) >= 4:
                    if abs(ratios[-1] - ratios[-2]) < 0.005 and abs(ratios[-2] - ratios[-3]) < 0.005 and abs(
                            ratios[-3] - ratios[-4]) < 0.005:
                        flag = True
                        break
                t += inc
            if flag:
                break
            inc /= 2
        return ratios[-1]

    def global_rate(self):
        # 根据信息获得合适的剪枝率——auto
        ratios = []

        for i in range(self.clients):
            ret = self.get_ratio(self.analyzer.user_groups[i])
            ratios.append(ret)

        # by la why multiply two rates
        for i in range(self.clients):
            self.global_prune_rate += self.rates[i] * ratios[i]
        return self.global_prune_rate

    def layer_rate(self):
        compress_rate = self.get_rate_for_each_layers()
        return compress_rate
