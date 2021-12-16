from abc import ABC, abstractmethod
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as tdata

from control.preEnv import *
import control.runtimeEnv as args
from model.vwrapper import VWrapper

class HRank(ABC):
    ERROR_MESS1 = "illegal mType(int)"
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    def __init__(self, mType: int, model: nn.Module) -> None:
        assert mType > ModelType.LOWWER.value, self.ERROR_MESS1
        assert mType < ModelType.UPPER.value, self.ERROR_MESS1
        self.mType = mType
        self.model = model

    #get feature map of certain layer via hook
    def get_feature_hook(self, module, input, output):
        imgs = output.shape[0]
        channels = output.shape[1]
        ranks = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() 
                            for i in range(imgs) for j in range(channels)])
        ranks = ranks.view(imgs, -1).float()
    
        # merge channel rank of all imgs
        ranks = ranks.sum(0)

        self.feature_result = self.feature_result * self.total + ranks
        self.total = self.total + imgs
        self.feature_result = self.feature_result / self.total

    @abstractmethod
    def feed_run(self, loader: tdata.DataLoader):
        pass

    @abstractmethod
    def get_rank(self, loader: tdata.DataLoader):
        pass

    @abstractmethod
    def get_mask(self):
        pass

class VGG16HRank(HRank):
    def __init__(self, mType: int, model: nn.Module) -> None:
        super().__init__(mType, model)
        self.wrapper = VWrapper(model)

    def get_rank(self, loader):
        if len(args.gpu) > 1:
            relucfg = self.model.module.relucfg
        else:
            relucfg = self.model.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = self.model.features[cov_id]
            handler = cov_layer.register_forward_hook(self.get_feature_hook)
            self.feed_run(loader)
            handler.remove()

            if not os.path.isdir('ranks/'+args.arch+'_limit%d'%(args.limit)):
                os.mkdir('ranks/'+args.arch+'_limit%d'%(args.limit))
            np.save('ranks/'+args.arch+'_limit%d'%(args.limit)+'/ranks' + str(i + 1) + '.npy', self.feature_result.numpy())

            self.feature_result = torch.tensor(0.)
            self.total = torch.tensor(0.)
    
    def get_mask(self):
        pass

    def feed_run(self, loader: tdata.DataLoader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = args.limit

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                #use the first 5 batches to estimate the rank.
                if batch_idx >= limit:
                    break

                loss, cort = self.wrapper.step_eva(inputs, targets)
                
                test_loss += loss
                correct += cort
                total += targets.size(0)
                
                GLOBAL_LOGGER.info(batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))