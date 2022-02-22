#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:49:21 2020

@author: gabby
"""
import numpy as np

G=6.67e-11 #mks
c=3*10**8 #mks
k=0 #curvature
h0=70 #Hubble constant, km/s/Mpc

om_bm0=0.022/(h0/100.0)**2 #Omega baryonic matter present
om_dm0=0.12/(h0/100.0)**2 #Omega dark matter present
om_l=0.69 #Omega lambda present
om_cmb=2.47e-5/(h0/100.)**2 #Omega CMB
om_nu=6e-4/(h0/100.0)**2 #Omega neutrinos lower limit
eps_c0=3*(h0*1000/(3.086e22))**2/(8*np.pi*G) #current critical energy density

h=3e14 #step size in eta, da=sqrt(8*pi*G*eps/3)*a^2*d(eta) from 1.70 Mukhanov
l=4830 #array size

a=np.zeros(l) #Scale factor
a[l-1]=1

eps_bm=np.zeros(l) #Baryons
eps_bm0=eps_c0*om_bm0
eps_bm[l-1]=eps_bm0

eps_dm=np.zeros(l) #Dark matter
eps_dm0=eps_c0*om_dm0
eps_dm[l-1]=eps_dm0

eps_cmb=np.zeros(l) #Cosmic Microwave Background photons
eps_cmb0=eps_c0*om_cmb
eps_cmb[l-1]=eps_cmb0

eps_l=np.zeros(l) #Lambda
eps_l0=eps_c0*om_l
eps_l[l-1]=eps_l0

eps=np.zeros(l) #Total energy
eps[l-1]=eps_dm0+eps_bm0+eps_cmb0+eps_l0

def ap(a, eps):
    return np.sqrt(8*np.pi*G*eps*a**4/3.0-k*a**2)

def ap2(a, epsbm, epsdm, epsl):
    return 4*np.pi*G*(epsdm+epsbm+4*epsl)*a**3/3-k*a

for i in np.arange(2,len(a)+1):
    y=a[l-i+1]
    a[l-i]=y-h*ap(y,eps[l-i+1]) #-0.5*h**2*ap2(y,eps_bm[l-i+1],eps_dm[l-i+1],eps_l[l-i+1])/2
    if i%100==0:
        print(y-h*ap(y,eps[l-i+1]))
    eps_bm[l-i]=eps_bm[l-i+1]*(a[l-i+1]/a[l-i])**3
    eps_dm[l-i]=eps_dm[l-i+1]*(a[l-i+1]/a[l-i])**3
    eps_cmb[l-i]=eps_cmb[l-i+1]*(a[l-i+1]/a[l-i])**4
    eps_l[l-i]=eps_l0
    eps[l-i]=eps_bm[l-i]+eps_dm[l-i]+eps_cmb[l-i]+eps_l[l-i]

