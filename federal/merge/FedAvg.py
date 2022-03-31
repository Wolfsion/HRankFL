from typing import List
from collections import OrderedDict

class FedAvg:
    ERROR_MESS1 = "The weights size must be equal the size of the clients_dicts"

    def __init__(self):
        pass

    def merge_dict(self, clients_dicts: List[dict], weights: List[int]) -> dict:
        assert len(clients_dicts) == len(weights), self.ERROR_MESS1
        sum_weight = sum(weights)
        proportion = [weight / sum_weight for weight in weights]
        merged_dict = OrderedDict()
        curt_idx = 0
        for dic in clients_dicts:
            for k, v in dic.items():
                if k in merged_dict.keys():
                    merged_dict[k] += v * proportion[curt_idx]
                else:
                    merged_dict[k] = v * proportion[curt_idx]
            curt_idx += 1
        return merged_dict

    def merge_info(self):
        pass
