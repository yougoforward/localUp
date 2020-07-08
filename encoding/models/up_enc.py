###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import encoding

from .base import BaseNet
from .fcn import FCNHead

__all__ = ['up_enc', 'EncModule', 'get_up_enc', 'get_up_enc_resnet50_pcontext',
           'get_up_enc_resnet101_pcontext', 'get_up_enc_resnet50_ade',
           'get_up_enc_resnet101_ade']

class up_enc(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(up_enc, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead([512, 1024, 2048], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
                            lateral=kwargs['lateral'], norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4, c20, c30, c40 = self.base_forward(x)
        x = list(self.head(c1,c2,c3,c4,c20,c30,c40))
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

        # self.localUp2=localUp(256, 512, norm_layer, up_kwargs)
        self.localUp3=localUp(512, 1024, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, 2048, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        out = self.localUp4(c3, c4)
        out = self.localUp3(c2, out)
        # out = self.localUp2(c1, out)
       
        feat = self.conv5(out)

        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)

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
                                   norm_layer(in_channels1//8))
        self.refine2 = nn.Sequential(nn.Conv2d(in_channels2, self.key_dim, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(in_channels1//8)) 
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2,out):
        n,c,h,w =c1.size()
        c1 = self.refine(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        c2 = self.refine2(c2)

        unfold_up_c2 = unfold(c2, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.matmul(c1.view(n, -1, 1, h*w).permute(0,3,2,1), unfold_up_c2.permute(0,3,1,2)) #n,h*w,1,3x3
        att = torch.softmax(energy, dim=-1)
        out = F.interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = unfold(out, 3, 2, 2, 1).view(n, -1, 3*3, h*w)
        out = torch.matmul(att, unfold_out.permute(0,3,2,1)).permute(0,3,2,1).view(n,-1,h,w)

        return out

def get_up_enc(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""up_enc model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    # infer number of classes
    from ..datasets import datasets
    model = up_enc(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('up_enc_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_up_enc_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_enc-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_up_enc('pcontext', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_up_enc_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_enc-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_up_enc('pcontext', 'resnet101', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, lateral=True, **kwargs)

def get_up_enc_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_enc-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_up_enc('ade20k', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_up_enc_resnet101_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_enc-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_up_enc('ade20k', 'resnet101', pretrained, root=root, aux=True,
                      base_size=640, crop_size=576, lateral=True, **kwargs)

def get_up_enc_resnet152_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""up_enc-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_up_enc_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_up_enc('ade20k', 'resnet152', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)
