import pickle
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

ranks_dict = OrderedDict()
num_clients = 100


def load():
    global ranks_dict
    with open('ranks', 'rb') as f:
        ranks_dict = pickle.load(f)


def view():
    layer_rank = []
    cos_ret = []
    for i in range(num_clients):
        rank = ranks_dict[i]
        for k, v in rank:
            print('---', k, '---')
            layer_rank.append(v)
            break
    master = layer_rank[num_clients - 1]
    for i in range(num_clients-1):
        cos_ret.append(cosine_similarity(master, layer_rank[i]))
    print(cos_ret)


def main():
    load()
    view()
