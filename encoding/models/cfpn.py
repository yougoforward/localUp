from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['cfpn', 'get_cfpn']


class cfpn(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(cfpn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = cfpnHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4, c20, c30, c40 = self.base_forward(x)
        x = self.head(c1,c2,c3,c4,c20,c30,c40)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)



class cfpnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(cfpnHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, 1, bias=True),
                            nn.Sigmoid())
        self.gff = PAM_Module(in_dim=inter_channels, key_dim=64,value_dim=inter_channels,out_dim=inter_channels,norm_layer=norm_layer)

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))

        self.localUp2=localUp2(256, 256, 256, norm_layer, up_kwargs)
        self.localUp3=localUp(512, inter_channels, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, inter_channels, norm_layer, up_kwargs)

        self.dconv4_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv4_8 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )

        self.dconv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv2_8 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv3_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv3_8 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.project4 = nn.Sequential(nn.Conv2d(2*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.project3 = nn.Sequential(nn.Conv2d(2*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.project2 = nn.Sequential(nn.Conv2d(2*inter_channels, 256, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU(),
                                   )
        self.project = nn.Sequential(nn.Conv2d(6*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        _,_, h,w = c2.size()
        # out4 = self.conv5(c4)
        p4_1 = self.dconv4_1(c4)
        p4_8 = self.dconv4_8(c4)
        out4 = self.project4(torch.cat([p4_1,p4_8], dim=1))

        out3 = self.localUp4(c3, out4)
        p3_1 = self.dconv3_1(out3)
        p3_8 = self.dconv3_8(out3)
        out3 = self.project3(torch.cat([p3_1,p3_8], dim=1))

        out2 = self.localUp3(c2, out3)
        p2_1 = self.dconv2_1(out2)
        p2_8 = self.dconv2_8(out2)
        # out2 = self.project2(torch.cat([p2_1,p2_8], dim=1))
        # #up 4
        # out2 = self.localUp2(c1, out2)

        p4_1 = F.interpolate(p4_1, (h,w), **self._up_kwargs)
        p4_8 = F.interpolate(p4_8, (h,w), **self._up_kwargs)
        p3_1 = F.interpolate(p3_1, (h,w), **self._up_kwargs)
        p3_8 = F.interpolate(p3_8, (h,w), **self._up_kwargs)
        out = self.project(torch.cat([p2_1,p2_8,p3_1,p3_8,p4_1,p4_8], dim=1))

        #gp
        gp = self.gap(c4)        
        # se
        se = self.se(gp)
        out = out + se*out

        #non-local
        # out = self.gff(out)


        out = torch.cat([out, gp.expand_as(out)], dim=1)

        return self.conv6(out)

class localUp(nn.Module):
    def __init__(self, in_channels, in_channels2, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())

        self._up_kwargs = up_kwargs
        self.refine = nn.Sequential(nn.Conv2d(in_channels2+out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(),
                                    )

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        out = torch.cat([c1,c2], dim=1)
        out = self.refine(out)
        return out

class localUp2(nn.Module):
    def __init__(self, in_channels, in_channels2, out_channels, norm_layer, up_kwargs):
        super(localUp2, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, 48, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(48),
                                   nn.ReLU())

        self._up_kwargs = up_kwargs
        self.refine = nn.Sequential(nn.Conv2d(in_channels2+48, out_channels, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(),
                                    )

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        out = torch.cat([c1,c2], dim=1)
        out = self.refine(out)
        return out

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    def __init__(self, inplanes, planes, outplanes, stride=1, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(
            planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
                nn.Conv2d(inplanes, outplanes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(outplanes),
            )
        self.dconv1 = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(planes),
                                   nn.ReLU(),
                                   )
        self.dconv2 = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=2, dilation=2, bias=False),
                                   norm_layer(planes),
                                   nn.ReLU(),
                                   )
        self.dconv3 = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=3, dilation=3, bias=False),
                                   norm_layer(planes),
                                   nn.ReLU(),
                                   )                                  
    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dconv1(out)
        # out2 = self.dconv2(out)
        # out3 = self.dconv3(out)
        
        # out = out1+out2+out3

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

def get_cfpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = cfpn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.pool = nn.AvgPool2d(kernel_size=2)

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

