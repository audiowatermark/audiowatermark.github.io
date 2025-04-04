#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:52:25 2020

@author: darp_lord
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class VGGM(nn.Module):

    def __init__(self, n_classes=10):
        super(VGGM, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=1)),
            # replace 7*7 to 3*3
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU()),
            # ('mpool1', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=1)),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU()),
            # ('mpool2', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('mpool5', nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9, 1), stride=(1, 1))),
            ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
            ('relu6', nn.ReLU()),
            ('apool6', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten())]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            # ('drop1', nn.Dropout()),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))

    def forward(self, inp):
        # [B, 225, T]
        inp = inp.transpose(1, 2)  # [B, T, 225]
        inp = inp.unsqueeze(1)  # [B, 1, T, 225]
        emb = self.features(inp)
        pred = self.classifier(emb)
        # print(emb.shape)
        # input: [2, 1, 160, 40] => 2: batch size, 160: frames, 40: mel-feature
        # output: [2, 4096]
        return pred, emb


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGM(567)
    model.to(device)
    print(summary(model, (1, 160, 40)))
