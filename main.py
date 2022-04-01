# from federal.simulation.simMain import main
# from federal.simulation.trainMain import main as train_main
from federal.simulation.suff_train import main
#from visual.test_unit import main
import sys

if __name__ == "__main__":
    # main()
    import torch
    import torchvision
    from dl.model import modelUtil
    vgg1 = torchvision.models.vgg16()
    dict1 = vgg1.state_dict()
    dict2 = vgg1.state_dict()
    modelUtil.dict_diff(dict1, dict2)
