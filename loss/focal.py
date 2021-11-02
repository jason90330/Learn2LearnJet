import torch
import torch.nn as nn
import numpy as np

# Support: ['FocalLoss']

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7, outNum = 6, withWeight = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps   

        self.gamma = gamma
        self.eps = eps        
        weight = torch.zeros(outNum).cuda()
        '''wrong(FP=197)
        weight[0]=0.2
        weight[1]=0.8
        '''
        
        #better(FP=142)
        weight[0]=0.33
        weight[1]=0.66

        
        '''wrong(FP=204)
        weight[0]=0.5
        weight[1]=0.5
        '''
        '''
        #wrong(FP=266)
        weight[0]=0.66
        weight[1]=0.33
        '''
        
        if withWeight:
            self.ce = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce = nn.CrossEntropyLoss()


    def forward(self, input, target):
        target = target.type(torch.int64)
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()