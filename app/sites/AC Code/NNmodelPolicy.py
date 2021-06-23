# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:05:42 2021

@author: bahramis
"""


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
        self.output1 = nn.Linear(hidden_layers[-1], output_size)
        # self.output2 = nn.Linear(hidden_layers[-1], output_size)
        # self.output3 = nn.Linear(hidden_layers[-1], 1)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    # def forwardMu(self, x):
    #     ''' Forward pass through the network, returns the output logits '''  
    #     for each in self.hidden_layers:
    #         x = self.LeakyReLU(each(x))
        
    #     x1 = self.output1(x)
    #     x2 = self.output2(x)
    #     # if torch.max(torch.abs(x))>35:
    #     #     x=(35*x/torch.max(torch.abs(x)))
    #     x1=self.output3(x1)
    #     x1=self.sigmoid(x1) 
    #     x2=self.softmax(x2) 
        
        # return  x1, x2
    
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''  
        for each in self.hidden_layers:
            x = self.Tanh(each(x))
        
        x = self.output1(x)
        if torch.max(torch.abs(x))>35:
            x=(35*x/torch.max(torch.abs(x)))
        x=self.softmax(x) 
        
        return  x

    # def forwardMu2(self, x):
    #     ''' Forward pass through the network, returns the output logits '''  
    #     for each in self.hidden_layers:
    #         x = self.LeakyReLU(each(x))
        
    #     x1 = self.output1(x)
    #     x2 = self.output2(x)
    #     # if torch.max(torch.abs(x))>35:
    #     #     x=(35*x/torch.max(torch.abs(x)))
    #     x1=self.softmax(x1) 
    #     x2=self.softmax(x2) 
        
        # return  x1, x2
    def forwardMudesc(self, x):
        ''' Forward pass through the network, returns the output logits '''
        for each in self.hidden_layers:
            x = F.leaky_relu(each(x))
        x = self.output(x)
        #   x =F.sigmoid(x)
        x=F.softmax(x)
        
        return x

    def forwardSigma(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = self.LeakyReLU(each(x))
        x = self.output(x)
        x = 10*F.softplus(x)
     #   x=F.softmax(x, dim=1)
        
        return x