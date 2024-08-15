#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMBH functions and DM mass function things from Anthony and Friends

Created on Tue Aug 13 15:12:11 2024

@author: gabbyhuckabee
"""

import constants as c
import numpy as np

ginf=1.43728 #Does Ginf need a conversion?  Seems dimensionless
xi2=10**(-17) #matter mass per photon (?) squared in 1/msol
rho_l=1.25e-123 #dark energy density in planck density
zeq=2740 #redshift of matter-radiation equality
xeq=c.om_l*(1+zeq)**(-3)/(c.om_bm0+c.om_dm0) #? this is rho_l/rho_m for rho_m at equality, see anthony's paper w max and frank
al=3215 #A_Lambda
q=2e-5 #Scalar fluctuation amplitude on horizon
gconst=0.652 #weak coupling constan at m_z
b=-0.27

def g(zed): #dimensionless
    mass=c.om_bm0+c.om_dm0
    x=(c.om_l/mass)*(1+zed)**(-3)
    gl = x**(1./3.)*(1+(x/(ginf**3))**(0.795))**(-1./(3.*0.795))
    gx = 1+3.*al*gl/2.
    return gx

def s(m): #dimensionless, checked for mu=1e-5 gives s=28
    mu = xi2*m #here use m in msol
    return ((9.1*mu**(-2./3))**b+(50.5*np.log10(834+mu**(-1./3.))-92)**b)**(1./b)

def sigma(sm, sz): #sz is redshift, sm is mass, this function is off
    return g(sz)*s(sm)*q

def f(m, z, epsbm, epsdm, epsc): #m in msol, this function works (matches Watson et al)
    om=(epsbm+epsdm)/epsc
    a2=om*(0.99*(1+z)**(-3.216)+0.074)
    alpha=om*(5.907*(1+z)**(-3.599)+2.344)
    beta=om*(3.136*(1+z)**(-3.058)+2.349)
    gamma=1.318
    return a2*((beta/sigma(m,z))**alpha+1)*np.exp(-gamma/sigma(m,z)**2)

def rhs(m, z, epsbm, epsdm, epsc, du): #m in msol
    rho=(epsbm+epsdm)/c.c**2
    temp=f(m,z,epsbm,epsdm,epsc)*rho/(m*c.msol)
    om_m=c.om_bm0+c.om_dm0
    dz=-(1+z)**2
    dx=c.om_l/om_m*(-3.)*(1+z)**(-4)*dz
    x=c.om_l/om_m*(1+z)**(-3)
    dgl=1./3.*x**(-2./3.)*(1+(x/ginf**3)**0.795)**(-1./(3.*0.795))*(1-x**0.795/(ginf**(3*0.795)*(1+(x/ginf**3)**0.795)))*dx
    mu=xi2*m
    ds=-du*((9.1*mu**(-2./3.))**b+(50.5*np.log10(834+mu**(-1./3.))-92)**b)**((1-b)/b)*((2./3.)*9.1**b*mu**(-(2*b-3)/3)+50.5*(50.5*np.log10(834+mu**(-1./3.))-92)**(b-1)/(3*mu**(4./3.)*(834+mu**(-1./3.))*np.log(10)))
    dlnsig=-q*(s(m)*(3./2.)*al*dgl+g(z)*ds)/sigma(m, z) #Just needs to be multiplied by da
    return temp*dlnsig #this, times da, will equal dn (number of halos between mass m and m+dm)