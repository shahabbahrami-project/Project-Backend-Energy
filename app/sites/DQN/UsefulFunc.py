# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:44:43 2019

@author: Shahabbahrami
"""
import numpy as np
from numpy.random import randint
from sites.DQN.PriceFunc import PriceFunc
def OutdoorTemp(t):
    if t%96<=48:
       Tout= 0.25*randint(56, 64)
    else:
       Tout= 0.25*randint(44, 52)
    return Tout

def People(t,FromHour,ToHour):
    if t%96<FromHour:
        N= 0
    elif t%96>=FromHour and t%96<=ToHour:
        N=  randint(1, 4)
    else:
        N=  0
    return N

def Price(t):
    D=96
    T=D*100
    PriceValue=np.zeros(T)
    PriceValue=np.tile(PriceFunc(D),int(T/D))
    return PriceValue[t]


def Desire(t, Desire,FromHour,ToHour):
    if t%96<FromHour:
        Tdes= 0
    elif t%96>=FromHour and t%96<=ToHour:
        Tdes=  Desire
    else:
        Tdes=  0
    return Tdes
