import pandas as pd
from fedlab.utils.dataset import CIFAR10Partitioner
from fedlab.utils.functional import partition_report
from matplotlib import pyplot as plt

from dl.data.dataProvider import get_data
from env.preEnv import DataSetType


def cifar10_100part_noniid():
    cifar10 = get_data(DataSetType.CIFAR10, data_type="train")
    hetero_dir_part = CIFAR10Partitioner(cifar10.targets, 100,
                                         balance=None, partition="dirichlet",
                                         dir_alpha=0.3, seed=2022)
    csv_file = "logs/cifar10_hetero_dir_0.3_100clients.csv"
    partition_report(cifar10.targets, hetero_dir_part.client_dict,
                     class_num=10,
                     verbose=False, file=csv_file)
    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(10)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.xlabel('sample num')
    plt.tight_layout()
    plt.savefig(f"res/images/cifar10_hetero_dir_0.3_10clients.png", dpi=400)

def cifar10_100part_noniid_shards():
    cifar10 = get_data(DataSetType.CIFAR10, data_type="train")
    num_shards = 200
    shards_part = CIFAR10Partitioner(cifar10.targets,
                                     100,
                                     balance=None,
                                     partition="shards",
                                     num_shards=num_shards,
                                     seed=2022)
    csv_file = "logs/cifar10_shards_dir_0.3_100clients.csv"
    partition_report(cifar10.targets, shards_part.client_dict,
                     class_num=10,
                     verbose=False, file=csv_file)
    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(10)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.xlabel('sample num')
    plt.tight_layout()
    plt.savefig(f"res/images/cifar10_shards_dir_0.3_10clients.png", dpi=400)

if __name__ == '__main__':
    cifar10_100part_noniid_shards()
