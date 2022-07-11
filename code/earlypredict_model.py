'''ConvNet-AIG in PyTorch.

Residual Network is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Adaptive Inference Graphs is from the original ConvNet-AIG paper:
[2] Andreas Veit, Serge Belognie
    Convolutional Networks with Adaptive Inference Graphs. ECCV 2018

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init

from gumbelmodule import GumbleSoftmax

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ExitBlock(nn.Module):
    def __init__(self, num_classes):
        super(ExitBlock, self).__init__()
        
        # Gate layers
        self.fc1 = nn.Linear(num_classes, num_classes)
        self.fc1bn = nn.BatchNorm1d(num_classes)
        self.fc2 = nn.Linear(num_classes, 2, bias=True)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.fc1bn(self.fc1(x)))
        mask = self.fc2(x)
        return mask


class EarlyPredict(nn.Module):
    def __init__(self, exit_num, num_classes=10):
        super(EarlyPredict, self).__init__()
        self.ExitGates = nn.ModuleList([ExitBlock(num_classes) for i in range(exit_num)])
        self.exit_num = exit_num
        self.gs = GumbleSoftmax()
        self.gs.cuda()
        self.apply(_weights_init)

    def forward(self, logits, temperature=1):
        assert len(logits) == self.exit_num +1, "input num {} != {}!!!".format(len(logits), self.exit_num +1)

        gs_masks = [op(x) for op,x in zip(self.ExitGates, logits[:self.exit_num])]
        masks = [self.gs(mask, temp=temperature, force_hard=True) for mask in gs_masks]

        masks.append(torch.ones(masks[0].size(), dtype = masks[0].dtype, device = masks[0].device, requires_grad = False))
        # import pdb;pdb.set_trace()
        flag = torch.ones(masks[0].size(), dtype = masks[0].dtype, device = masks[0].device, requires_grad = False)
        for i in range(1,self.exit_num + 1):
            flag = flag * (1-masks[i-1])
            masks[i] = masks[i] * flag

        new_logit = sum([x * m[:,1].view(-1, 1) for m,x in zip(masks, logits)])

        assert torch.sum(torch.stack(masks)[:,:,1]) ==  logits[0].size(0), "early prediction mask error!!!"
        return new_logit, [mask[:,1] for mask in masks]  # 1exit  0continue


def EP_ResNet_cifar(nclass=10):
    return EarlyPredict(exit_num = 2, num_classes=nclass)


def EP_ResNet_ImageNet():
    return EarlyPredict(exit_num = 3, num_classes=1000)


# EP = EP_ResNet110_cifar(nclass=10)
# x = torch.rand(16,10)
# x, masks = EP([x,x,x])
# print(masks)