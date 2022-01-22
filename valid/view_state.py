import pickle
from collections import OrderedDict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ranks_dict = OrderedDict()
num_clients = 100


def load():
    global ranks_dict
    with open('ranks.ret', 'rb') as f:
        ranks_dict = pickle.load(f)


def view():
    layer_rank = [[] for _ in range(len(ranks_dict[0]))]
    for i in range(num_clients):
        rank = ranks_dict[i]
        for index, val in enumerate(rank.values()):
            layer_rank[index].append(np.array(val))
    for i in range(len(layer_rank)):
        print(cosine_similarity(np.array(layer_rank[i])))


def main():
    load()
    view()
