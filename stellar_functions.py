#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy from starlight and dust reprocessing functions

Created on Fri Feb 14 2025

@author: gabbyhuckabee
"""

import constants as c
import numpy as np

rsol = 6.9566e8 #Ekers 2018, m
gmsol = 1.3271244e20 #Ekers 2018, m^3/s^2

#Peak SFR is around 5-6 z
def strdot(h): #Time derivative of stellar mass density, h is Hubble constant in (km/s)/Mpc
    x=(h/c.h0)**(2./3.)
    sol=c.strdot0*x**2/(1+0.012*(x-1)**3*np.exp(0.041*x**(7./4.))) #Hernquist & Springel
    return sol #d(rho)/dt in (kg/m^3)/s

def imf(m): #Initial mass function, redshift independent, m in msol
    if m<0.5:
        alpha=-1.35
        x_err=0.5
    elif m<1:
        alpha=-2.35
        x_err=0.3
    else:
        alpha=-2.35
        x_err=0.7
    return m**(alpha+1), x_err*m**(alpha+1)*np.log(1/m) #dn/dlogM, error of dn/dlogM

def norm(epsstr): #normalization constant between rho_str(z)*1 million yrs (1 star formation event) and imf/pmf integral
    sol=1.70939 #Integrated in Mathematica from 0 msol to 300 msol
    a=epsstr/(sol*c.msol*c.c**2)
    return a #1/volume

#Stellar mass (in Msol) to lifetime
def tstar(m):
    #sol=10**(10)*m**(-2.5)
    sol=10**(10.015-3.461*np.log10(m)+0.8157*(np.log10(m)**2)) #elmegreen 2007 paper, eq2
    return sol

def luminosity(m): #m in msol
    if 0.179<m<=0.45:
        logL = 2.028*np.log10(m)-9.76
    elif 0.45<m<=0.72:
        logL = 4.572*np.log10(m)-0.102
    elif 0.72<m<=1.05:
        logL = 5.743*np.log10(m)-0.007
    elif 1.05<m<=2.40:
        logL = 4.329*np.log10(m)+0.010
    elif 2.40<m<=7:
        logL = 3.976*np.log10(m)-0.093
    else: #this function only reliably models up to m=31
        logL = 2.865*np.log10(m)-1.105
    L = 10**(logL)
    return L #L in Lsol

def temp(m): #m in msol
    if m<=1.5: #only reliable down to m=0.179
        R = (0.438*m**2+0.479*m+0.075)*rsol
        logT = luminosity(m)*lsol/(4*np.pi*R**2*sigmasb)**(1./4.)
    else: #only reliable up to m=31
        logT = -0.170*np.log10(m)**2+0.888*np.log10(m)+3.671
    T = 10**(logT)
    return T #K

def entropyrate(m): #time derivative of entropy for one star of mass m
    return luminosity(m)*lsol/temp(m) #proportional to ds/dt

"""
s per timestep = number density of stars of mass m*(ds(m)/dt)*dt
s per timestep = integrate IMF*dlogM*(ds(m)/dt)*dt over logM
"""