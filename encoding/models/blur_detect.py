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

__all__ = ['blur_detect', 'get_blur_detect']

class blur_detect(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(blur_detect, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = blur_detectHead(2048, nclass, norm_layer, self._up_kwargs)
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


        
class blur_detectHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(blur_detectHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels*2, out_channels, 1))

        self.localUp2=localUp(256, in_channels, norm_layer, up_kwargs)
        self.localUp3=localUp(512, in_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, in_channels, norm_layer, up_kwargs)

        self.denseaspp = DenseASPP(inter_channels, inter_channels, norm_layer)
        self.pam0 = PAM_Module(in_dim=inter_channels, key_dim=inter_channels//8, value_dim=inter_channels, out_dim=inter_channels, norm_layer=norm_layer)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, 1, bias=True),
                            nn.Sigmoid())

    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        out = self.localUp4(c3, c4)
        out = self.localUp3(c2, out)
        # out = self.localUp2(c1, out)
        out = self.denseaspp(out)

        #gp
        gp = self.gap(c4)
        # se
        se = self.se(gp)
        out = out + se*out
        # gff
        out = self.pam0(out)
        n,c,h,w=out.size()
        out = torch.cat([out, gp.expand(n,c,h,w)], dim=1)
        return self.conv6(out)

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

class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DenseASPP, self).__init__()
        inter_channels = in_channels // 4

        self.ASPP_3 = ASPPConv(in_channels, out_channels, 3, norm_layer)

        self.ASPP_6 = ASPPConv(in_channels+out_channels, out_channels, 6, norm_layer)
        
        self.ASPP_12 = ASPPConv(in_channels+out_channels*2, out_channels, 12, norm_layer)

        self.ASPP_18 = ASPPConv(in_channels+out_channels*3, out_channels, 18, norm_layer)

        self.ASPP_24 = ASPPConv(in_channels+out_channels*4, out_channels, 24, norm_layer)
        self.project = nn.Sequential(nn.Conv2d(in_channels+out_channels*5, out_channels, 1, padding=0, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(),
                                   )

    def forward(self, feature):
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        out = self.project(feature)
        return out

def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = inter_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        # self.project = nn.Sequential(
        #     nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=value_dim, out_channels=value_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)
        # self.fuse_conv = nn.Sequential(nn.Conv2d(value_dim, out_dim, 1, bias=False),
        #                                norm_layer(out_dim),
        #                                nn.ReLU(True))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        xp = self.pool(x)
        m_batchsize, C, height, width = x.size()
        m_batchsize, C, hp, wp = xp.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(xp).view(m_batchsize, -1, wp*hp)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        proj_value = xp.view(m_batchsize, -1, wp*hp)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # out = F.interpolate(out, (height, width), mode="bilinear", align_corners=True)

        gamma = self.gamma(x)
        out = (1-gamma)*out + gamma*x
        # out = self.fuse_conv(out)
        return out
def get_blur_detect(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""blur_detect model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_blur_detect.pdf>`_
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
    >>> model = get_blur_detect(dataset='pascal_voc', backbone='resnet50', pretrained=False)
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
    model = blur_detect(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('blur_detect_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


