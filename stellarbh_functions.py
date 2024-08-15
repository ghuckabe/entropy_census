#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stellar mass black hole number density calculations
Uses Chabrier IMF, assume black hole progenitors are greater than 25 Msol
Progenitor to remnant mass function crudely approximated from Fryer & Kalogera, 2001
https://arxiv.org/pdf/astro-ph/9911312.pdf

Created on Tue Aug 13 15:00:47 2024

@author: gabbyhuckabee
"""

import constants as c
import numpy as np

#Peak SFR is around 5-6 z
def strdot(h): #Time derivative of stellar mass density, h is Hubble constant in (km/s)/Mpc
    x=(h/c.h0)**(2./3.)
    sol=c.strdot0*x**2/(1+0.012*(x-1)**3*np.exp(0.041*x**(7./4.))) #Hernquist & Springel
    return sol #d(rho)/dt in (kg/m^3)/s

def prog2rem(m): #Progenitor mass to remnant mass, function approximated from plot in Fryer & Kalogera 2001, m in msol
    if m<20:
        sol=1.2
    elif m>42:
        sol=m
    else:
        sol=0.625*m-11.25
    return sol #returns BH mass in msol

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

def imfhigh(m): #Initial mass function, redshift independent, m in msol
    if m<0.5:
        alpha=-(1.35+0.5) #pm 0.5
    elif m<1:
        alpha=-(2.35-0.3) #pm 0.3
    else:
        alpha=-(2.35-0.7) #pm 0.7
    return m**(alpha+1) #multiply by dlogM to return value proportional to dn in 

def imflow(m): #initial mass function redshift independent, m in msol
    if m<0.5:
        alpha=-(1.35-0.5)
    elif m<1:
        alpha=-(2.35+0.3)
    else:
        alpha=-(2.35+0.7)
    return m**(alpha+1) #multiply by dlogM to get proportional to dn

#Reproduce Fig 14 in Sicilia 2022 to check functions (and check log vs ln)
# z_scale=np.array([0,1,2,4,6,8,10])
# fit_n_array=np.exp(mpc**(-3)*np.array([5.623, 5.429, 5.107, 4.344, 3.614, 2.894, 2.305])) #Mpc^-3
# fit_m_array=np.exp(msol*np.array([0.607, 0.609, 0.612, 0.634, 0.659, 0.676, 0.680]))
# alpha_array=np.array([-3.781, -3.859, -3.914, -3.902, -3.866, -3.868, -3.884])
# fit_ng_array=np.exp(mpc**(-3)*np.array([2.413, 2.309, 2.064, 1.419, 0.806, 0.197, -0.344]))
# fit_mG_array=np.exp(msol*np.array([2.021, 2.023, 2.024, 2.037, 2.054, 2.066, 2.072]))
# sigma_G_array=np.array([0.052, 0.051, 0.051, 0.049, 0.045, 0.043, 0.042])

z_scale=np.array([0,1,2,4,6,8,10])
fit_n_array=10**(np.array([5.623, 5.429, 5.107, 4.344, 3.614, 2.894, 2.305])) #Mpc^-3
fit_m_array=10**(np.array([0.607, 0.609, 0.612, 0.634, 0.659, 0.676, 0.680]))
alpha_array=np.array([-3.781, -3.859, -3.914, -3.902, -3.866, -3.868, -3.884])
fit_ng_array=10**(np.array([2.413, 2.309, 2.064, 1.419, 0.806, 0.197, -0.344]))
fit_mG_array=10**(np.array([2.021, 2.023, 2.024, 2.037, 2.054, 2.066, 2.072]))
sigma_G_array=np.array([0.052, 0.051, 0.051, 0.049, 0.045, 0.043, 0.042])

def dn_sicilia(m, z): #m in msol, z is redshift
    #m=m*msol
    fit_n=np.interp(z, z_scale, fit_n_array)
    fit_m=np.interp(z,z_scale, fit_m_array)
    alpha=np.interp(z,z_scale, alpha_array)
    fit_ng=np.interp(z,z_scale, fit_ng_array)
    fit_mG=np.interp(z,z_scale, fit_mG_array)
    sigma_G=np.interp(z,z_scale, sigma_G_array)
    # fit_n=fit_n_array[0]
    # fit_m=fit_m_array[0]
    # alpha=alpha_array[0]
    # fit_ng=fit_ng_array[0]
    # fit_mG=fit_mG_array[0]
    # sigma_G=sigma_G_array[0]
    return c.mpc**(-3)*(fit_n*(m/fit_m)**(1-alpha)*np.exp(-m/fit_m)+fit_ng*np.exp(-((np.log(m)-np.log(fit_mG))**2)/(2*sigma_G**2))/np.sqrt(2*np.pi*sigma_G**2))#multiply by dlogM to get dn

def norm(epsstr): #normalization constant between rho_str(z)*1 million yrs (1 star formation event) and imf/pmf integral
    sol=1.70939 #Integrated in Mathematica from 0 msol to 300 msol
    a=epsstr/(sol*c.msol*c.c**2)
    return a #1/volume

def normhigh(epsstr):
    solhigh=3.08867 #from +--
    ahigh=epsstr/(solhigh*c.msol*c.c**2) 
    return ahigh

def normlow(epsstr):
    sollow=1.18153 #from -++
    alow=epsstr/(sollow*c.msol*c.c**2)
    return alow

#Stellar mass (in Msol) to lifetime

def tstar(m):
    #sol=10**(10)*m**(-2.5)
    sol=10**(10.015-3.461*np.log10(m)+0.8157*(np.log10(m)**2)) #elmegreen 2007 paper, eq2
    return sol
