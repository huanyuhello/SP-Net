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

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Sequential_ext(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """
    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, temperature=1, openings=None):
        gate_activations = []
        gate_logits = []
        # feature = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation, gate_logit = module(input, temperature)
            gate_activations.append(gate_activation)
            gate_logits.append(gate_logit)
            # feature.append(input)
        return input, gate_activations, gate_logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = LambdaLayer(lambda x:
                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion*planes)
            # )
        # Gate layers
        # self.fc1 = nn.Conv2d(in_planes, 128, kernel_size=1)
        self.fc1 = nn.Linear(in_planes, 128)
        self.fc1bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2, bias=True)
        self.dropout = nn.Dropout(p=0.4)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1):
        # Compute relevance score
        w = F.avg_pool2d(x, x.size(2)).squeeze()
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.dropout(w)

        gate_logit = self.fc2(w)

        # Sample from Gumble Module
        w = self.gs(gate_logit, temp=temperature, force_hard=True)


        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out * w[:,1].view(-1, 1, 1, 1)
        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        # return out, w[:, 1], F.softmax(gate_logit, 1)
        return out, w[:, 1], gate_logit



class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.auxiliary1 = nn.AvgPool2d(32, 32)
        self.auxiliary2 = nn.AvgPool2d(16, 16)
        self.auxiliary3 = nn.AvgPool2d(8, 8)

        self.pool_list = nn.ModuleList([self.auxiliary1, self.auxiliary2, self.auxiliary3])

        self.fc1 = nn.Linear(16 * block.expansion, num_classes)
        self.fc2 = nn.Linear(32 * block.expansion, num_classes)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, x, temperature=1, openings=None):
        gate_activations = []
        feature_list = []
        gate_logits = []
        block_feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out, a, g, f = self.layer1(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        block_feature_list.append(f)

        out, a, g, f= self.layer2(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        block_feature_list.append(f)

        out, a, g, f = self.layer3(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        block_feature_list.append(f)
        # print([block_feature_list[0][i].shape for i in range(18)])
        # print([block_feature_list[1][i].shape for i in range(18)])
        # print([block_feature_list[2][i].shape for i in range(18)])

        diff_list = []
        for stage_i in range(3):
            stage_f = block_feature_list[stage_i]
            f0 = self.pool_list[stage_i](stage_f[0]).view(out.size(0), -1)
            f1 = self.pool_list[stage_i](stage_f[int(len(stage_f)//2)]).view(out.size(0), -1)
            f2 = self.pool_list[stage_i](stage_f[-1]).view(out.size(0), -1)
            diff0 = f1-f0 
            diff1 = f2-f1
            diff_list.append([diff0, diff1])
        # import pdb; pdb.set_trace()
        out1_feature = self.auxiliary1(feature_list[0]).view(out.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(out.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(out.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.linear(out3_feature)

        return [out3, out2, out1], diff_list , gate_activations, gate_logits


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        self.fc1 = nn.Linear(in_planes, 128)
        self.fc1bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2, bias=True)
        self.dropout = nn.Dropout(p=0.4)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2

        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1):
        # Compute relevance score

        ######### batch size >1 
        w = F.avg_pool2d(x, x.size(2)).squeeze()
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.dropout(w)
        gate_logit = self.fc2(w)
        # Sample from Gumble Module
        w = self.gs(gate_logit , temp=temperature, force_hard=True)

        # TODO: For fast inference, check decision of gate and jump right 
        #       to the next layer if needed.

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.downsample(x) + out * w[:,1].view(-1, 1, 1, 1)
        out = F.relu(out, inplace=True)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, w[:, 1], gate_logit


class ResNet_ImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.auxiliary1 = nn.AvgPool2d((56, 56))
        self.auxiliary2 = nn.AvgPool2d((28, 28))
        self.auxiliary3 = nn.AvgPool2d((14, 14))
        self.auxiliary4 = nn.AvgPool2d((7, 7))

        # self.c_p = nn.AdaptiveAvgPool1d(64)
        self.cc1 = nn.Linear(64 * block.expansion, 2048)
        self.cc2 = nn.Linear(128 * block.expansion, 2048)
        self.cc3 = nn.Linear(256 * block.expansion, 2048)

        self.pool_list = nn.ModuleList([self.auxiliary1, self.auxiliary2, self.auxiliary3, self.auxiliary4])


        # self.fc1 = nn.Linear(64 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(128 * block.expansion, num_classes)
        # self.fc3 = nn.Linear(256 * block.expansion, num_classes)
        self.fc  = nn.Linear(512 * block.expansion, num_classes)

        self.fc1 = nn.Linear( 2048, num_classes)
        self.fc2 = nn.Linear( 2048, num_classes)
        self.fc3 = nn.Linear( 2048, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if 'fc2' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, out, temperature=1):
        gate_activations = []
        feature_list = []
        gate_logits = []

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        # print('layer1:', out.shape)
        out, a, g = self.layer1(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        # print('layer2:', out.shape)
        out, a, g = self.layer2(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        # print('layer3:', out.shape)
        out, a, g = self.layer3(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)
        # print('layer4:', out.shape)
        out, a, g = self.layer4(out, temperature)
        gate_activations.extend(a)
        gate_logits.extend(g)
        feature_list.append(out)


        out1_feature = self.auxiliary1(feature_list[0]).view(out.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(out.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(out.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[3]).view(out.size(0), -1)

        # f1 = F.interpolate(out1_feature.unsqueeze(1), size=[2048]).squeeze()
        # f2 = F.interpolate(out2_feature.unsqueeze(1), size=[2048]).squeeze()
        # f3 = F.interpolate(out3_feature.unsqueeze(1), size=[2048]).squeeze()
        # diff0 = f2-f1
        # diff1 = f3-f2
        # diff2 = f4-f3

        f1 = self.cc1(out1_feature)
        f2 = self.cc2(out2_feature)
        f3 = self.cc3(out3_feature)
        f4 = out4_feature
        diff0 = f4-f1
        diff1 = f4-f2
        diff2 = f4-f3

        out1 = self.fc1(f1)
        out2 = self.fc2(f2)
        out3 = self.fc3(f3)
        out4 = self.fc(f4)

        return [out4, out3, out2, out1], [diff0, diff1, diff2], gate_activations, gate_logits




def DR_ResNet110_cifar(nclass=10):
    return ResNet_cifar(BasicBlock, [18,18,18], num_classes=nclass)

def DR_ResNet32_cifar(nclass=10):
    return ResNet_cifar(BasicBlock, [5,5,5], num_classes=nclass)



def DR_ResNet50_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,4,6,3])

def DR_ResNet101_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,4,23,3])

def DR_ResNet152_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,8,36,3])

