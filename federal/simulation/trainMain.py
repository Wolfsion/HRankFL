from federal.FLnodes import *
from federal.simulation.trainCVH import RunPlus


def main():
    dic = {"cs": 10}
    args = argparse.Namespace(**dic)
    fl_runner = RunPlus(args)
    for i in range(10):
        fl_runner.upload_download_dict()
    fl_runner.valid_acc()


