import argparse
from abc import ABC, abstractmethod
class Parser(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def autoParse(self, json):
        pass

class RankGenParser(Parser):
    def parse(self):
        parser = argparse.ArgumentParser(description='Rank extraction')
        parser.add_argument(
            '--data_dir',
            type=str,
            default='./data',
            help='dataset path')
        parser.add_argument(
            '--dataset',
            type=str,
            default='cifar10',
            choices=('cifar10','imagenet'),
            help='dataset')
        parser.add_argument(
            '--job_dir',
            type=str,
            default='result/tmp',
            help='The directory where the summaries will be stored.')
        parser.add_argument(
            '--arch',
            type=str,
            default='vgg_16_bn',
            choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','mobilenet_v2','mobilenet_v1'),
            help='The architecture to prune')
        parser.add_argument(
            '--pretrain_dir',
            type=str,
            default=None,
            help='load the model from the specified checkpoint')
        parser.add_argument(
            '--limit',
            type=int,
            default=5,
            help='The num of batch to get rank.')
        parser.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help='Batch size for training.')
        parser.add_argument(
            '--gpu',
            type=str,
            default='0',
            help='Select gpu to use')

        return parser.parse_args()
    
    def autoParse(self, json):
        
        pass
        
class PruningFLParser(Parser):
    def parse(self):
        parser = argparse.ArgumentParser()
        mutex = parser.add_mutually_exclusive_group(required=True)
        mutex.add_argument('-a', '--adaptive',
                        help="Use adaptive compress",
                        action='store_true',
                        dest='use_adaptive')
        mutex.add_argument('-na', '--no-adaptive',
                        help="Do not use adaptive compress",
                        action='store_false',
                        dest='use_adaptive')

        mutex1 = parser.add_mutually_exclusive_group(required=True)
        mutex1.add_argument('-i', '--init-compress',
                            help="Use initial compress",
                            action='store_true',
                            dest='initial_pruning')
        mutex1.add_argument('-ni', '--no-init-compress',
                            help="Do not use initial compress",
                            action='store_false',
                            dest='initial_pruning')

        parser.add_argument('-c', '--client-selection',
                            help="If use client-selection",
                            action='store_true',
                            dest='client_selection',
                            default=False,
                            required=False)
        parser.add_argument('-t', '--target-density',
                            help="Target density",
                            action='store',
                            dest='target_density',
                            type=float,
                            required=False)
        parser.add_argument('-m', '--max-density',
                            help="Max density",
                            action='store',
                            dest='max_density',
                            type=float,
                            required=False)
        parser.add_argument('-s', '--seed',
                            help="The seed to use for the prototype",
                            action='store',
                            dest='seed',
                            type=int,
                            default=0,
                            required=False)
        parser.add_argument('-e', '--exp-name',
                            help="Experiment name",
                            action='store',
                            dest='experiment_name',
                            type=str,
                            required=True)
        return parser.parse_args()

class GlobalYMLParser(Parser):
    pass
