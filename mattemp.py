#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:06:43 2020

@author: gabby
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit as fit


z=[0.07,0.11,0.14,0.18,0.22,0.27,0.31,0.36,0.4,0.45,0.5,0.55,0.61,0.66,0.72,0.82,0.98,1.15,1.33,1.53,1.75,1.98,2.24,2.52,2.82,3.15]
a=[]
for i in z:
    a.append(1.0/i)

te=[2.51,1.47,1.52,1.93,2.07,0.69,1.4,0.79,1.92,1.33,1.15,1.56,0.77,1.16,1.26,0.91,0.31,1.55,0.61,0.37,0.42,0.54,0.17,0.18,0.27,0.32]
t=[]
for tmp in te:
    t.append(tmp*10**6)

#plt.scatter(a,t)

def func(x,a,b,k):
    return a*np.log10(b*x)+k

popt,pcov=fit(func,a,t)

plt.plot(a,func(a,*popt))