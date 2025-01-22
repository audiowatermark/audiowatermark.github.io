#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn
from hparam import hparam as hp
from torchsummary import summary


class SpeechEmbedder(nn.Module):

    def __init__(self, n_classes=10):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)
        self.classifier = nn.Linear(hp.model.proj, n_classes)

    def forward(self, x):
        # [B, 225, T]
        x = x.transpose(1, 2)
        # [B, T, 225]
        x, _ = self.LSTM_stack(x.float()) 
        # [B, T, hidden]
        x = x[:, x.size(1) - 1]
        # [B, hidden]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        pred = self.classifier(x)

        return pred, _


