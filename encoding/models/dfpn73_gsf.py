from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['dfpn73_gsf', 'get_dfpn73_gsf']


class dfpn73_gsf(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(dfpn73_gsf, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = dfpn73_gsfHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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



class dfpn73_gsfHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(dfpn73_gsfHead, self).__init__()
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
        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)

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
        # out = self.localUp2(c1, out)
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
        out = self.gff(out)

        out = torch.cat([out, gp.expand_as(out)], dim=1)

        return self.conv6(out)

class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())

        self._up_kwargs = up_kwargs
        # self.refine = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
        #                            norm_layer(out_channels),
        #                            nn.ReLU(),
        #                             )
        self.refine = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(),
                                    )
        # self.refine = Bottleneck(inplanes = 2*out_channels, planes=2*out_channels//4, outplanes=out_channels, stride=1, dilation=1, norm_layer=norm_layer)
    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        out = torch.cat([c1,c2], dim=1)
        out = self.refine(out)
        return out


def get_dfpn73_gsf(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = dfpn73_gsf(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1,2,3,6), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        priors.append(self.pool(feats).view(n, c, -1))
        center = torch.cat(priors, -1)
        return center
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer, psp_size=(1,2,3,6)):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.psp = PSPModule(psp_size)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        xp = self.psp(x)
        m_batchsize, C, height, width = x.size()
        m_batchsize, C, hpwp = xp.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(xp)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = xp
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        gamma = self.gamma(x)
        out = (1-gamma)*out + gamma*x
        return out
# class PAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.pool = nn.MaxPool2d(kernel_size=2)

#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
#         self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         xp = self.pool(x)
#         m_batchsize, C, height, width = x.size()
#         m_batchsize, C, hp, wp = xp.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(xp).view(m_batchsize, -1, wp*hp)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#         proj_value = xp.view(m_batchsize, -1, wp*hp)
        
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#         # out = F.interpolate(out, (height, width), mode="bilinear", align_corners=True)

#         gamma = self.gamma(x)
#         out = (1-gamma)*out + gamma*x
#         # out = self.fuse_conv(out)
#         return out

# class PSPModule(nn.Module):
#     # (1, 2, 3, 6)
#     def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
#         super(PSPModule, self).__init__()
#         self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

#     def _make_stage(self, size, dimension=2):
#         if dimension == 1:
#             prior = nn.AdaptiveAvgPool1d(output_size=size)
#         elif dimension == 2:
#             prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         elif dimension == 3:
#             prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
#         return prior

#     def forward(self, feats):
#         n, c, _, _ = feats.size()
#         priors = [stage(feats).view(n, c, -1) for stage in self.stages]
#         center = torch.cat(priors, -1)
#         return center

# class APAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer, psp_size=(1,3,6,8)):
#         super(APAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.key_channels = key_dim
#         self.psp = PSPModule(psp_size)
#         self.query_conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1, bias=False),
#             norm_layer(key_dim),
#             nn.ReLU(True))
#         self.key_conv = self.query_conv
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=value_dim, kernel_size=1)
#         self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

#         self.softmax = nn.Softmax(dim=-1)
#         self.fuse_conv = nn.Sequential(nn.Conv2d(value_dim, out_dim, 1, bias=False),
#                                        norm_layer(out_dim),
#                                        nn.ReLU(True))

#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.psp(self.key_conv(x))
#         energy = torch.bmm(proj_query, proj_key)

#         energy = (self.key_channels ** -.5) * energy
#         attention = self.softmax(energy)
#         proj_value = self.psp(self.value_conv(x))
        
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, -1, height, width)
        
#         out =self.fuse_conv(out)
#         gamma = self.gamma(x)
#         out = (1-gamma)*out + gamma*x
#         return out

class CLF_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer, up_kwargs):
        super(CLF_Module, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=value_dim, kernel_size=1)

        self._up_kwargs = up_kwargs

    def forward(self, x, coarse):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n,c,h,w = x.size()
        ncls = coarse.size()[1]
        coarse = F.interpolate(coarse, (h,w), **self._up_kwargs)
        coarse = coarse.view(n, ncls, -1).permute(0,2,1)
        coarse_norm = F.softmax(coarse, dim=1)
        class_feat = torch.matmul(x.view(n,c,-1), coarse_norm) # n x c x ncls


        proj_query = self.query_conv(x).view(n, -1, h*w).permute(0, 2, 1)
        proj_key = self.key_conv(class_feat)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(n, -1, h*w)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(n, c, h, w)
        return out