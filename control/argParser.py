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
        