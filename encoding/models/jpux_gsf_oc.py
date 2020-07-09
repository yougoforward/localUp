from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['jpux_gsf_oc', 'get_jpux_gsf_oc']


class jpux_gsf_oc(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(jpux_gsf_oc, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = jpux_gsf_ocHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        # _, _, c3, c4 = self.base_forward(x)
        # x = list(self.head(c4))

        c1, c2, c3, c4, c20, c30, c40 = self.base_forward(x)
        x = list(self.head(c1,c2,c3,c4,c20,c30,c40))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        x[1] = F.interpolate(x[1], (h, w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class jpux_gsf_ocHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(jpux_gsf_ocHead, self).__init__()
        inter_channels = in_channels // 4
        self.jpu = JPU_X([512, 1024, 2048], width=inter_channels, norm_layer=norm_layer, up_kwargs=up_kwargs)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=inter_channels*4, out_channels=inter_channels, kernel_size=1, padding=0, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        # self.conv52 = nn.Sequential(nn.Conv2d(in_channels=inter_channels*4, out_channels=inter_channels, kernel_size=1, padding=0, bias=False),
        #                     norm_layer(inter_channels),
        #                     nn.ReLU(True))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, 1, bias=True),
                            nn.Sigmoid())
        self.gff = PAM_Module(in_dim=inter_channels, key_dim=inter_channels//8,value_dim=inter_channels,out_dim=inter_channels,norm_layer=norm_layer)

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))
        self.ocr = ocr_Module(in_dim=inter_channels, key_dim=inter_channels//8,value_dim=inter_channels,out_dim=out_channels,norm_layer=norm_layer)
    def forward(self, c1,c2,c3,c4,c20,c30,c40):
        x = c4
        c1, c2, c3, c4 = self.jpu(c1, c2, c3, c4)
        out = self.conv5(c4)
        #gp
        gp = self.gap(x)        
        # se
        se = self.se(gp)
        out = out + se*out

        #non-local
        out = self.gff(out) #n,c,h,w

        # object context 
        out, sigmoid_pred, center_pred =self.ocr(out, gp)
        return self.conv6(out), sigmoid_pred, center_pred

class ocr_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
        super(ocr_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.conv62 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*in_dim, out_dim, 1))
        self.nclass = out_dim
        self.dataset_centers = nn.Parameter(torch.zeros(in_dim, self.nclass))
        self.center_pred = nn.Sequential(nn.Dropout2d(0.1), nn.Conv1d(in_dim, out_dim, 1))

        self.project = nn.Sequential(nn.Conv2d(in_channels=in_dim*2, out_channels=value_dim, kernel_size=1, padding=0, bias=False),
                            norm_layer(value_dim),
                            nn.ReLU(True))
    def forward(self, x, gp):
        out = torch.cat([x, gp.expand_as(x)], dim=1)
        # object context dictionary
        sigmoid_pred = self.conv62(out)
        sigmoid_att = torch.sigmoid(sigmoid_pred) #n,nclass,h,w
        # object center
        n,c,h,w = x.size()
        att = sigmoid_att.view(n,self.nclass, h*w).permute(0,2,1) # n,hw, nclass
        image_centers = torch.bmm(x.view(n, c, h*w), att/torch.sum(att, 1, keepdim=True).expand_as(att)) #n, c, nclass
        class_centers = torch.cat([self.dataset_centers.expand_as(image_centers), image_centers], dim=1) # n, c, 2*nclass

        proj_query = self.query_conv(x).view(n, -1, h*w).permute(0, 2, 1)
        proj_key = self.key_conv(class_centers).view(n,-1, self.nclass*2)
        energy = torch.bmm(proj_query, proj_key) #n, h*w, 2*nclass
        attention = self.softmax(energy)
        proj_value = torch.bmm(class_centers, attention.permute(0,2,1)).view(n, c, h, w)
        
        out = self.project(torch.cat([x, proj_value], dim=1))

        out = torch.cat([out, gp.expand_as(x)], dim=1)
        return out, sigmoid_pred ,self.center_pred(self.dataset_centers.unsqueeze(0))


def gsnetConv(in_channels, out_channels, atrous_rate, norm_layer):
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


def get_jpux_gsf_oc(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = jpux_gsf_oc(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
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



class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat


class JUM(nn.Module):
    def __init__(self, in_channels, width, dilation, norm_layer, up_kwargs):
        super(JUM, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        norm_layer = lambda n_channels: nn.GroupNorm(32, n_channels)
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=dilation, dilation=dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2*dilation, dilation=2*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=4*dilation, dilation=4*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, x_l, x_h):
        feats = [self.conv_l(x_l), self.conv_h(x_h)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([feats[-2], self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)

        return feat

class JPU_X(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU_X, self).__init__()
        self.jum_1 = JUM(in_channels[:2], width//2, 1, norm_layer, up_kwargs)
        self.jum_2 = JUM(in_channels[1:], width, 2, norm_layer, up_kwargs)

    def forward(self, *inputs):
        feat = self.jum_1(inputs[2], inputs[1])
        feat = self.jum_2(inputs[3], feat)

        return inputs[0], inputs[1], inputs[2], feat
