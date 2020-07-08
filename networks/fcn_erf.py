import os
import sys
from inplace_abn import InPlaceABN, InPlaceABNSync
from networks.cc_edge import CCEdgeGuide
from utils.pyt_utils import load_model
from utils.pyt_utils import decode_labels,decode_edge_labels,decode_logits,decode_edge_logits
import functools
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import wandb

affine_par = True


BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.seg_dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

        # edge case
        self.edge_refine = CCEdgeGuide()
        self.edge_side5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=True))

        self.criterion = criterion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        def generate_multi_grid(index, grids): return grids[index % len(
            grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation,
                            downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def config_network(self, edge_branch='none', seg_branch='none', circumular='gradual'):
        if edge_branch == 'none':
            self.edge_forward = False
        elif edge_branch == 'fix':
            for name, params in self.named_parameters():
                if 'edge' in name:
                    params.requires_grad = False
        else:
            self.edge_forward = True
        if seg_branch is None:
            self.seg_forward = False
        elif seg_branch == 'fix':
            for name, params in self.named_parameters():
                if 'seg' in name:
                    params.requires_grad = False
        else:
            self.seg_forward = True

        if (not self.edge_forward) and (not self.seg_forward):
            raise TypeError("Should either branch be activated")
        if circumular == 'gt':
            self.ratio = 0
            self.delta_ratio = 0
        elif circumular == 'gradual':
            self.ratio = 0
            self.delta_ratio = 0.001
        else:
            self.ratio = 1
            self.delta_ratio = 0
        self.vis = False
        self.logger = None 

    def config_logger(self,logger):
        self.logger=logger
        self.global_step = -1

    def forward(self, x, labels=None, edges=None):
        imgs = x 
        x = self.relu1(self.bn1(self.conv1(imgs)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        outs = []
        edge_out = []
        edge_for_refine = []
        if self.seg_forward:
            x_dsn = self.seg_dsn(x)
            outs.append(x_dsn)
        x = self.layer4(x)
        if self.edge_forward:
            edge_out.append(self.edge_side5(x))

        if self.seg_forward:
            x_before_refine = self.seg_head(x)
            outs.append(x_before_refine)

        if self.seg_forward and self.edge_forward:
            # dilate or not

            self.ratio = min(self.ratio, 1)
            edge_for_refine = edge_out[-1]*self.ratio+F.adaptive_avg_pool2d(
                edges, x_before_refine.size()[-2:])*edge_out[-1].max()*(1-self.ratio)

            x_after_refine = self.edge_refine(x_before_refine, edge_for_refine)
            edge_for_refine = [edge_for_refine]
            self.ratio += self.delta_ratio

            outs.append(x_after_refine)

        if self.criterion is not None and labels is not None:
            loss = self.criterion(outs, labels, edge_out, edges)
            if self.vis:
                self.display([imgs,outs, labels, edge_out+edge_for_refine, edges])
                self.criterion.show_detail()
            return loss

        else:
            return outs[::-1]

    def display(self, flag=None,global_step=-1):
        if flag is None:
            self.vis = True
            self.global_step=global_step

        else:
            # visualize
            if self.logger is not None:
                self.logger.visualize(*flag)
            self.vis = False
def Seg_Model(num_classes, criterion=None, pretrained_model=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, criterion)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
        # device = torch.device('cpu')
        # saved_state_dict = torch.load(pretrained_model, map_location=device)
        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     #Scale.layer5.conv2d_list.3.weight
        #     i_parts = i.split('.')
        #     # print i_parts
        #     # if not i_parts[1]=='layer5':
        #     if not i_parts[0]=='fc':
        #         new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        # model.load_state_dict(new_params)

    return model
