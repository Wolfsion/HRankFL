from random import random

from utils.DataExtractor import Extractor
from utils.Visualizer import VisBoard


def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list


def get_lists():
    lists = [[range(100), random_list(), random_list()]]
    return lists


def main():
    repo = Extractor()
    vis = VisBoard(repo)
    vis.single_var_dist("F", "k")
    #vis.double_vars_dist("F", "kr")
    #vis.double_vars_regression("F", "kr")


if __name__ == '__main__':
    repo = Extractor()
    vis = VisBoard(repo)
    vis.single_var_dist("F", "kr")
