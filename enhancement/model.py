import torch
import torch.nn as nn
# from loss import LossFunction
import os
import time
import random
import tqdm

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta

class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        # self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    # def _loss(self, input):
    #     i_list, en_list, in_list, _ = self(input)
    #     loss = 0
    #     for i in range(self.stage):
    #         loss += self._criterion(in_list[i], i_list[i])
    #     return loss

class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        # self._criterion = LossFunction()

        if weights is not None:
            base_weights = torch.load(weights)
            pretrained_dict = base_weights
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    # def _loss(self, input):
    #     i, r = self(input)
    #     loss = self._criterion(input, i)
    #     return loss


# class DecomNet(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(DecomNet, self).__init__()
#         # Shallow feature extraction
#         self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
#                                     padding=4, padding_mode='replicate')
#         # Activated layers!
#         self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
#                                                   padding=1, padding_mode='replicate'),
#                                         nn.ReLU(),
#                                         nn.Conv2d(channel, channel, kernel_size,
#                                                   padding=1, padding_mode='replicate'),
#                                         nn.ReLU(),
#                                         nn.Conv2d(channel, channel, kernel_size,
#                                                   padding=1, padding_mode='replicate'),
#                                         nn.ReLU(),
#                                         nn.Conv2d(channel, channel, kernel_size,
#                                                   padding=1, padding_mode='replicate'),
#                                         nn.ReLU(),
#                                         nn.Conv2d(channel, channel, kernel_size,
#                                                   padding=1, padding_mode='replicate'),
#                                         nn.ReLU())
#         # Final recon layer
#         self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
#                                     padding=1, padding_mode='replicate')

#     def forward(self, input_im):
#         input_max= torch.max(input_im, dim=1, keepdim=True)[0]
#         input_img= torch.cat((input_max, input_im), dim=1)
#         feats0   = self.net1_conv0(input_img)
#         featss   = self.net1_convs(feats0)
#         outs     = self.net1_recon(featss)
#         R        = torch.sigmoid(outs[:, 0:3, :, :])
#         L        = torch.sigmoid(outs[:, 3:4, :, :])
#         return R, L
    

class DecomNet(nn.Module):
    def __init__(self, channel=8, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(3, channel, kernel_size,
                                    padding=1, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 3, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        feats0   = self.net1_conv0(input_im)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        outs     = torch.clamp(outs, 0.001, 1)
        return outs  

