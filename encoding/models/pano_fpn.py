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

__all__ = ['pano_fpn', 'get_pano_fpn']

class pano_fpn(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(pano_fpn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = pano_fpnHead(2048, nclass, norm_layer, self._up_kwargs)
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

        
class pano_fpnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(pano_fpnHead, self).__init__()
        inter_channels = in_channels // 8
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels//2, out_channels, 1))

        self.localUp2=localUp(256, inter_channels, norm_layer, up_kwargs)
        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)
        self.refine = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self._up_kwargs = up_kwargs
        self.seg = pfpn(inter_channels, inter_channels//2, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        out32 = self.conv5(c4)
        out16 = self.localUp4(c3, out32)
        out8 = self.localUp3(c2, out16)
        out4 = self.localUp2(c1, out8)

        # out = self.refine(out4)
        out = self.seg(out4, out8, out16, out32)
        
        return self.conv6(out)

class pfpn(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(pfpn, self).__init__()
        self._up_kwargs = up_kwargs
        inter_channels = out_channels
        self.p32_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.p32_conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.p32_conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.p16_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.p16_conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.p8_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.p4_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
    def forward(self, c1,c2,c3,c4):
        _,_,h1,w1 =c1.size()
        _,_,h2,w2 =c2.size()
        _,_,h3,w3 =c3.size()
        
        p32 = self.p32_conv1(c4)
        p32 = interpolate(p32, (h3,w3), **self._up_kwargs)
        p32 = self.p32_conv2(p32)
        p32 = interpolate(p32, (h2,w2), **self._up_kwargs)
        p32 = self.p32_conv3(p32)
        p32 = interpolate(p32, (h1,w1), **self._up_kwargs)

        p16 = self.p16_conv1(c3)
        p16 = interpolate(p16, (h2,w2), **self._up_kwargs)
        p16 = self.p16_conv2(p16)
        p16 = interpolate(p16, (h1,w1), **self._up_kwargs)

        p8 = self.p8_conv1(c2)
        p8 = interpolate(p8, (h1,w1), **self._up_kwargs)

        p4 = self.p4_conv1(c1)

        out = p32 + p16 + p8 + p4
        return out

class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = interpolate(c2, (h,w), **self._up_kwargs)
        out = c1+c2
        return out


def get_pano_fpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""pano_fpn model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_pano_fpn.pdf>`_
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
    >>> model = get_pano_fpn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
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
    model = pano_fpn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('pano_fpn_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


