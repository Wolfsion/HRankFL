from abc import ABC, abstractmethod
import numpy as np
import os
import torch
import torch.nn as nn

from control.preEnv import *
import control.runtimeEnv as args


criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)

#get feature map of certain layer via hook


def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use the first 5 batches to estimate the rank.
            if batch_idx >= limit:
               break

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''

class HRank(ABC):
    ERROR_MESS1 = "illegal mType(int)"
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    def __init__(self, mType, model) -> None:
        assert mType > ModelType.LOWWER, self.ERROR_MESS1
        assert mType < ModelType.UPPER, self.ERROR_MESS1
        self.mType = mType
        self.model = model

    def get_feature_hook(self, input, output):
        imgs = output.shape[0]
        channels = output.shape[1]
        ranks = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() 
                            for i in range(imgs) for j in range(channels)])
        ranks = ranks.view(imgs, -1).float()
    
        # merge channel rank of all imgs
        ranks = ranks.sum(0)

        feature_result = self.feature_result * self.total + ranks
        total = self.total + imgs
        feature_result = feature_result / total

    @abstractmethod
    def getRank():
        pass
    @abstractmethod
    def getMask():
        pass

class VGG16HRank(HRank):
    def getRank(self):
        pass
    def getMask(self):
        pass

    def refernece(self):
        if len(args.gpu) > 1:
            relucfg = self.model.module.relucfg
        else:
            relucfg = self.model.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = self.model.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            if not os.path.isdir('rank_conv/'+args.arch+'_limit%d'%(args.limit)):
                os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
            np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)