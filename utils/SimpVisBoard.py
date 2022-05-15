import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

DATA1 = "res/exp/resnet18_base.txt"
DATA2 = "res/exp/resnet18_pr.txt"
CLASS1 = "resnet18-base"
CLASS2 = "resnet18-prune"
IMG = r"res/images/resnet18_base.png"


class SVisBoard:
    def __init__(self):
        self.dataframe = None

    def parse(self):
        with open(DATA1, 'r') as f:
            content = f.read()
        acc_list1 = content.split(",")
        acc_list1 = list(map(float, acc_list1))
        total = len(acc_list1)
        round_list = list(range(1, total + 1))
        class_list1 = [CLASS1 for _ in range(total)]
        dic1 = {'round': round_list, 'acc': acc_list1, 'class': class_list1}
        df1 = pd.DataFrame(dic1)

        with open(DATA2, 'r') as f:
            content = f.read()
        acc_list2 = content.split(",")
        map(float, acc_list2)
        class_list2 = [CLASS2 for _ in range(total)]
        dic2 = {'round': round_list, 'acc': acc_list2, 'class': class_list2}
        df2 = pd.DataFrame(dic2)
        self.dataframe = pd.concat([df1, df2], ignore_index=True)

        # self.dataframe = df1

    def create(self):
        sns.set_style("darkgrid")
        tmp = sns.lineplot(data=self.dataframe, x="round",
                     y="acc",
                     ci=None, hue="class")
        tmp.invert_yaxis()
        plt.title('Top1-Acc~Round')
        plt.savefig(IMG)

