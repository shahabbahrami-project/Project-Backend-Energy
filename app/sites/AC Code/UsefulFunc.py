# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:44:43 2019

@author: Shahabbahrami
"""
import numpy as np
from scipy.stats import truncnorm
import torch
import NNmodelPolicy as nnPolicy
import NNmodelValue as nnValue
import math



# Load Neural Network Model
def load_checkpointPolicy(filepath):
    checkpoint = torch.load(filepath)
    model = nnPolicy.Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
def load_checkpointValue(filepath):
    checkpoint = torch.load(filepath)
    model = nnValue.Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Truncated Normal Distribution 
def Trunc(low, up, mu, sigma):
    a, b = (low - mu) / sigma, (up - mu) / sigma
    r = truncnorm.rvs(a, b, scale=sigma, loc=mu, size=1)
    return r

def criterionvalue(Vnew, Vold, cost, beta,delta):
    return (cost+beta*Vnew-Vold).pow(2)

#def criterionpolicy(action, mu, delta):
#    return -delta*(action-mu).pow(2)
def criterionpolicy(action, mu, sigma2, delta):
    cost1=(((action-mu).pow(2))/(2*sigma2))+torch.log(torch.sqrt(2*math.pi*sigma2))
    cost2=torch.sqrt(2*math.pi*sigma2)*torch.exp(((action-mu).pow(2))/(2*sigma2))
    return -delta*cost1*cost2

#def criterionpolicyDes(ps, delta):
#    return -delta*torch.sum(torch.mul(ps,torch.log(ps)))

def criterionpolicyDes(ps,pa,psvec,pavec,delta):
    return delta*(torch.log(ps)+torch.log(pa))
def criterionpolicyCom(pa,pavec,delta):
    return delta*(torch.log(pa))
#def criterionpolicyDes(ps,psvec,delta):
#    x=np.arange(0,1,0.111)
#    y=torch.from_numpy(x).float()
#    return delta*torch.sum(torch.mul(psvec,y))

#-0.1*torch.sum(psvec*torch.log(psvec))