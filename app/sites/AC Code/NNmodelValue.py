# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:02:54 2019

@author: Shahabbahrami
"""

import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.Tanh = nn.Tanh()
        
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        for each in self.hidden_layers:
            x = self.Tanh(each(x))
        x = self.output(x)
        return x

