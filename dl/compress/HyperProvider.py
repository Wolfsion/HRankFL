from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import comb
from env.preEnv import GLOBAL_LOGGER


class IntervalProvider:
    SIMP_LEN = 2
    ERROR_MESS1 = "Call is_timing_simple, the len of ranks_dict list must be 2."

    def __init__(self, tiny_const: int = 0.15, sum_limit: int = 0.1):
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

        GLOBAL_LOGGER.info(f"Sum distance:{sum_distance}")
        if sum_distance > combine_num * num_conv_layers * self.sum_limit:
            return False
        else:
            return True

    def is_timing_simple(self, ranks_dict: list = None):
        if ranks_dict is None:
            ranks_dict = self.cont_list
        assert len(ranks_dict) == self.SIMP_LEN, self.ERROR_MESS1
        return self.is_timing(ranks_dict, self.SIMP_LEN)

    def push_simp_container(self, ranks: dict):
        if len(self.cont_list) < 2:
            self.cont_list.append(ranks)
        else:
            self.cont_list[0] = self.cont_list[1]
            self.cont_list[1] = ranks

class RateProvider:
    def __init__(self):
        pass
