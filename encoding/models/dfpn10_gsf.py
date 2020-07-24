from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['dfpn10_gsf', 'get_dfpn10_gsf']


class dfpn10_gsf(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(dfpn10_gsf, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = dfpn10_gsfHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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



class dfpn10_gsfHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(dfpn10_gsfHead, self).__init__()
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
        self.gff = PAM_Module(in_dim=inter_channels, key_dim=inter_channels//8,value_dim=inter_channels,out_dim=inter_channels,norm_layer=norm_layer)

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))

        # self.localUp2=localUp(256, in_channels, norm_layer, up_kwargs)
        # self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        # self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)

        self.localUp3=localUp(512, 1024, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, 2048, norm_layer, up_kwargs)
        self.dconv1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.dconv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=8, dilation=8, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.project = nn.Sequential(nn.Conv2d(4*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        _,_, h,w = c2.size()
        out4 = self.conv5(c4)
        # out3 = self.localUp4(c3, out4)
        # out2 = self.localUp3(c2, out3)

        out3 = self.localUp4(c3, c40, out4)
        out2 = self.localUp3(c2, c30, out3)

        p4 = self.dconv4(out4)
        p4 = F.interpolate(p4, (h,w), **self._up_kwargs)
        p3 = self.dconv3(out3)
        p3 = F.interpolate(p3, (h,w), **self._up_kwargs)
        p2 = self.dconv2(out2)
        p1 = self.dconv1(out2)

        out = self.project(torch.cat([p1,p2,p3,p4], dim=1))
        #gp
        gp = self.gap(c4)        
        # se
        se = self.se(gp)
        out = out + se*out

        #non-local
        out = self.gff(out)

        out = torch.cat([out, gp.expand_as(out)], dim=1)

        return self.conv6(out)

# class localUp(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
#         super(localUp, self).__init__()
#         self.connect = nn.Sequential(
#                                    nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
#                                    norm_layer(out_channels),
#                                    nn.ReLU())


#         self._up_kwargs = up_kwargs



#     def forward(self, c1,c2):
#         n,c,h,w =c1.size()
#         c1 = self.connect(c1) # n, 64, h, w
#         c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
#         out = c1+c2
#         return out
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
        self.refine = nn.Sequential(nn.Conv2d(in_channels1, self.key_dim, 1, padding=0, dilation=1, bias=False))
        self.refine2 = nn.Sequential(nn.Conv2d(in_channels2, self.key_dim, 1, padding=0, dilation=1, bias=False)) 
        self._up_kwargs = up_kwargs
        self.unfold = nn.Unfold(3, 2, 2, 1)



    def forward(self, c1,c2,out):
        n,c,h,w =c1.size()
        c1 = F.normalize(self.refine(c1)) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        c2 = F.normalize(self.refine2(c2))

        unfold_up_c2 = self.unfold(c2).view(n, self.key_dim, 3*3, h, w)
        # torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
        energy = torch.einsum('nchw, nckhw -> nkhw', c1, unfold_up_c2)
        att = torch.softmax(energy, dim=1)
        out = F.interpolate(out, (h,w), **self._up_kwargs)
        unfold_out = self.unfold(out).view(n, -1, 3*3, h, w)
        out = torch.einsum('nkhw, nckhw -> nchw', att, unfold_out)
        return out

def get_dfpn10_gsf(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = dfpn10_gsf(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
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