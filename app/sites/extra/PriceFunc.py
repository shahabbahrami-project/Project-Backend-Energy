# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:13:49 2019

@author: Shahabbahrami
"""
import numpy as np
#def PriceFunc(D):
#    Price=np.zeros(D)
#    for i in range(0, D):
#        if i>=0 and i<=11:
#            Price[i]=5
#        if i>=12 and i<=23:
#            Price[i]=3
#        if i>=24 and i<=35:
#            Price[i]=5;
#        if i>=36 and i<=47:
#            Price[i]=7;
#        if i>=48 and i<=59:
#            Price[i]=10;
#        if i>=60 and i<=71:
#            Price[i]=10;
#        if i>=72 and i<=83:
#            Price[i]=7;
#        if i>=84 and i<=96:
#            Price[i]=3;
#    return Price



def PriceFunc(D):
    Price=np.zeros(D)
    for i in range(0, D):
        if i>=0 and i<=23:
            Price[i]=0.5
        if i>=24 and i<=35:
            Price[i]=0.25;
        if i>=36 and i<=47:
            Price[i]=0.6;
        if i>=48 and i<=59:
            Price[i]=1;
        if i>=60 and i<=83:
            Price[i]=0.5;
        if i>=84 and i<=96:
            Price[i]=0.25;
    return Price
#
#
#def PriceFunc(D):
#    Price=np.zeros(D)
#    for i in range(0, D):
#        if i>=0 and i<=0:
#            Price[i]=10
#        if i>=1 and i<=2:
#            Price[i]=1
#        if i>=3 and i<=4:
#            Price[i]=10
#    return Price
