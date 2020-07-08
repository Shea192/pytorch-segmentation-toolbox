import torch.nn as nn
import math
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .loss import OhemCrossEntropy2d
from .lovasz_losses import lovasz_softmax
from .loss import WeightedBinaryCrossEntropy2d
import scipy.ndimage as nd

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4


class CriterionOhemDSN2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        loss2 = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

        return loss1 + loss2

class CriterionEdgeWithSeg(nn.Module):
    def __init__(self, seg_weight='0.4,0.4,1',edge_weight='1',ignore_index=255, use_weight=True, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.edge_criterion = WeightedBinaryCrossEntropy2d()
        if not reduction:
            print("disabled the reduction.")
        self.seg_weight = [float(w) for w in seg_weight.split(',')]
        self.edge_weight = [float(w) for w in edge_weight.split(',')]
        print('Create loss function :\n Seg Loss Weight: {}\n Edge Loss Weight: {}\n'.format(self.seg_weight, self.edge_weight))
        self.loss_dict=dict()
    def forward(self, seg_preds, seg_target,edge_preds,edge_target):
        h, w = seg_target.size(1), seg_target.size(2)

        seg_weight = self.seg_weight 
        if len(seg_weight)<len(seg_preds): seg_weight = seg_weight+[1]*(len(seg_preds)-len(seg_weight))
        loss = 0
        i = 0
        for pred,weight in zip(seg_preds,seg_weight):
            scale_pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
            uloss=weight*self.seg_criterion(scale_pred,seg_target)
            loss+= uloss
            self.loss_dict['seg_'+str(i)+'_%.2f'%weight]=uloss.item()
            i+=1 
        
        edge_weight = self.edge_weight
        if len(edge_weight)<len(edge_preds): edge_weight = edge_weight + [1]*(len(edge_preds)-len(edge_weight))

        i=0
        for pred,weight in zip(edge_preds,edge_weight):
            scale_pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
            uloss = weight*self.edge_criterion(scale_pred,edge_target)        
            loss+= uloss

            self.loss_dict['edge_'+str(i)+'_%.2f'%weight]=uloss.item()
            i+=1 
#        self.show_detail()
        return loss
    def show_detail(self):
        string='\n'
        for k,v in self.loss_dict.items():
            string+='{}:{} '.format(k,v)
        print(string)
