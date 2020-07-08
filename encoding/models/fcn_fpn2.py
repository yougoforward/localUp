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

__all__ = ['fcn_fpn2', 'get_fcn_fpn2']

class fcn_fpn2(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(fcn_fpn2, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = fcn_fpn2Head(2048, nclass, norm_layer, self._up_kwargs)
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

        
class fcn_fpn2Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(fcn_fpn2Head, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

        self.localUp2=localUp(256, inter_channels, norm_layer, up_kwargs)
        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)
        self.refine2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.refine3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.conv_down = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, stride=2, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.project = nn.Sequential(nn.Conv2d(inter_channels*4, inter_channels, 1, padding=0, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self._up_kwargs = up_kwargs

    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        _,_,h,w = c2.size()
        p5 = self.conv_down(c4)
        p5 = interpolate(p5, (h,w), **self._up_kwargs)
        p4 = self.conv5(c4)
        p4 = interpolate(p4, (h,w), **self._up_kwargs)

        out3 = self.localUp4(c3, p4)
        out2 = self.localUp3(c2, out3)
        # out = self.localUp2(c1, out)
        p3 = self.refine3(out3)
        p3 = interpolate(p3, (h,w), **self._up_kwargs)
        p2 = self.refine2(out2)

        out = self.project(torch.cat([p5,p4,p3,p2], dim=1))
        return self.conv6(out)

class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        out = c1+c2
        return out


def get_fcn_fpn2(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""fcn_fpn2 model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn_fpn2.pdf>`_
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
    >>> model = get_fcn_fpn2(dataset='pascal_voc', backbone='resnet50', pretrained=False)
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
    model = fcn_fpn2(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_fpn2_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


