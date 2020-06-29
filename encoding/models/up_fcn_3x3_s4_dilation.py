###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import interpolate, unfold
from .fcn import FCNHead

from .base import BaseNet

__all__ = ['up_fcn_3x3_s4_dilation', 'get_up_fcn_3x3_s4_dilation']

class up_fcn_3x3_s4_dilation(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(up_fcn_3x3_s4_dilation, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = up_fcn_3x3_s4_dilationHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)


    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4, c20, c30, c40 = self.base_forward(x)
        x = self.head(c1,c2,c3,c4,c20,c30,c40)

        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

        
class up_fcn_3x3_s4_dilationHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(up_fcn_3x3_s4_dilationHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

        self.localUp2=localUp2(256, 512, norm_layer, up_kwargs)
        self.localUp3=localUp(512, 1024, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, 2048, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        out = self.conv5(c4)
        out = self.localUp4(c3, c40, out)
        out = self.localUp3(c2, c30, out)
        out = self.localUp2(c1, c2, out)
        return self.conv6(out)

class localUp(nn.Module):
    def __init__(self, in_channels1, in_channels2, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.key_dim = in_channels1//8
        # self.refine = nn.Sequential(nn.Conv2d(256, 64, 3, padding=2, dilation=2, bias=False),
        #                            norm_layer(64),
        #                            nn.ReLU(),
        #                            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
        #                            norm_layer(64),
        #                            nn.ReLU())
        self.refine = nn.Sequential(nn.Conv2d(in_channels1, self.key_dim, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(self.key_dim))
        self.refine2 = nn.Sequential(nn.Conv2d(in_channels2, self.key_dim, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(self.key_dim)) 
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2,out):
        n,c,h,w =c1.size()
        c1 = self.refine(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        c2 = self.refine2(c2)

        unfold_up_c2 = unfold(c2, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.matmul(c1.view(n, -1, 1, h*w).permute(0,3,2,1), unfold_up_c2.permute(0,3,1,2)) #n,h*w,1,3x3
        att = torch.softmax(energy, dim=-1)
        out = interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = unfold(out, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        out = torch.matmul(att, unfold_out.permute(0,3,2,1)).permute(0,3,2,1).view(n,-1,h,w)

        return out
class localUp2(nn.Module):
    def __init__(self, in_channels1, in_channels2, norm_layer, up_kwargs):
        super(localUp2, self).__init__()
        self.key_dim = in_channels1//4
        # self.refine = nn.Sequential(nn.Conv2d(256, 64, 3, padding=2, dilation=2, bias=False),
        #                            norm_layer(64),
        #                            nn.ReLU(),
        #                            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
        #                            norm_layer(64),
        #                            nn.ReLU())
        self.refine = nn.Sequential(nn.Conv2d(in_channels1, self.key_dim, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(self.key_dim))
        self.refine2 = nn.Sequential(nn.Conv2d(in_channels2, self.key_dim, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(self.key_dim)) 
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2,out):
        n,c,h,w =c1.size()
        c1 = self.refine(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        c2 = self.refine2(c2)

        unfold_up_c2 = unfold(c2, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.matmul(c1.view(n, -1, 1, h*w).permute(0,3,2,1), unfold_up_c2.permute(0,3,1,2)) #n,h*w,1,3x3
        att = torch.softmax(energy, dim=-1)
        out = interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = unfold(out, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        out = torch.matmul(att, unfold_out.permute(0,3,2,1)).permute(0,3,2,1).view(n,-1,h,w)

        return out
def get_up_fcn_3x3_s4_dilation(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""up_fcn_3x3_s4_dilation model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_up_fcn_3x3_s4_dilation.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_up_fcn_3x3_s4_dilation(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = up_fcn_3x3_s4_dilation(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('up_fcn_3x3_s4_dilation_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


