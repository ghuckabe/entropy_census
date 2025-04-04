#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:14:21 2024

@author: gabbyhuckabee

Late universe arrays

Initialize arrays for scale factor, energy density, entropy density, temperature
"""

import constants as c
import numpy as np

time=np.zeros(c.l) #Cosmic time
time[-1]=c.t0

eta=np.zeros(c.l) #Conformal time
eta[-1]=c.eta0

a=np.zeros(c.l) #Scale factor
a[-1]=1
a_ceharg=2982

hub=np.zeros(c.l) #Hubble constant
hub[-1]=c.h0

r_p=np.zeros(c.l) #Particle Horizon (Observable Universe)
r_p[-1]=c.eta0*c.c

volume=np.zeros(c.l) #Volume uses particle horizon
volume[-1]=4*np.pi*(r_p[-1])**3/3

r_ceh=np.zeros(c.l) #Radius of Cosmic Event Horizon
r_ceh[-1]=c.c*(c.etainf-c.eta0)
r_ceh_h=np.zeros(c.l) #Radius of Cosmic Event Horizon
r_ceh_h[-1]=c.c*(c.etainf_h-c.eta0_h)
r_ceh_l=np.zeros(c.l) #Radius of Cosmic Event Horizon
r_ceh_l[-1]=c.c*(c.etainf_l-c.eta0_l)

volumeceh=np.zeros(c.l) #Volume calculated with CEH
volumeceh[-1]=4*np.pi*(r_ceh[-1])**3/3
volumeceh_h=np.zeros(c.l)
volumeceh_h[-1]=4*np.pi*(r_ceh_h[-1])**3/3
volumeceh_l=np.zeros(c.l)
volumeceh_l[-1]=4*np.pi*(r_ceh_l[-1])**3/3

s_ceh=np.zeros(c.l)
stot_ceh=np.zeros(c.l)
s_ceh[-1]=(r_ceh[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh[-1]
stot_ceh[-1]=(r_ceh[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
s_ceh_h=np.zeros(c.l)
stot_ceh_h=np.zeros(c.l)
s_ceh_h[-1]=(r_ceh_h[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_h[-1]
stot_ceh_h[-1]=(r_ceh_h[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
s_ceh_l=np.zeros(c.l)
stot_ceh_l=np.zeros(c.l)
s_ceh_l[-1]=(r_ceh_l[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_l[-1]
stot_ceh_l[-1]=(r_ceh_l[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))

eps_bm=np.zeros(c.l) #Baryons
eps_bm0=c.eps_c0*c.om_bm0
eps_bm[-1]=eps_bm0
t_bm=np.zeros(c.l)
t_bm[-1]=0.025468041207723818 #This came from old ASU code from Judd Bowman?
s_bm=np.zeros(c.l)

#Eagle data
bm_eagleresults=[148.4053081304242, 84.3039732671589, 418.17238398806165, 853.2427715295006, 1572.864365719125, 1810.0005155900126, 1759.1301851009557, 2699.516883529645, 2110.990845022963, 2226.961377352639, 3126.284474107758, 4430.343761110442, 6127.405729410453, 4862.621531291029, 7113.963501171243, 8153.852287086782, 6080.247274201631, 8615.433172615767, 1927.9765390416364, 8000.553030185015, 1305.644296343521, 2968.6629285091212, 824.4928913315442, 1269.6081292868707, 464.85246944704045, 280.57777072388836, 233.8795437672676, 232.7706791309389, 235.5692882942148]
bmz=np.array([0.00, 0.10, 0.18, 0.27, 0.37, 0.50, 0.62, 0.74, 0.87, 1.00, 1.26, 1.49, 1.74, 2.01, 2.24, 2.48, 3.02, 3.53, 3.98, 4.49, 5.04, 5.49, 5.97, 7.05, 8.07, 8.99, 9.99, 15.13, 20.00])
bma=1/(bmz+1)
s_bm[-1]=bm_eagleresults[0]*c.kb

eps_str=np.zeros(c.l) #Stars
eps_str0=c.om_str0*c.eps_c0
eps_str[-1]=eps_str0
s_str=np.zeros(c.l)

eps_bh=np.zeros(c.l) #Stellar mass black holes
eps_bh0=0.56*10**8*c.msol*c.c**2/c.mpc**3 #Fukugita & Peebles
eps_bh[-1]=eps_bh0
s_bh=np.zeros(c.l)
s_bh0=1.6*10**(17)*c.kb #Egan and Lineweaver 2009
s_bh[-1]=s_bh0
s_bhhigh=np.zeros(c.l)
s_bhlow=np.zeros(c.l)
s_bh_err=np.zeros(c.l)
s_bh_sicilia=np.zeros(c.l)

eps_dm=np.zeros(c.l) #Dark matter
eps_dm0=c.eps_c0*c.om_dm0
eps_dm[-1]=eps_dm0
t_dm=np.zeros(c.l)
s_dm=np.zeros(c.l)

eps_remn=np.zeros(c.l) #White dwarves, neutron stars, stellar mass black holes
eps_remn0=c.eps_c0*c.om_remn
eps_remn[-1]=eps_remn0
t_remn=np.zeros(c.l)
s_remn=np.zeros(c.l)

eps_cmb=np.zeros(c.l) #Cosmic Microwave Background photons
eps_cmb0=c.eps_c0*c.om_cmb
eps_cmb[-1]=eps_cmb0
t_cmb=np.zeros(c.l)
t_cmb0=2.7255 #K, Fixen 2009
t_cmb[-1]=t_cmb0
s_cmb=np.zeros(c.l)
s_cmb[-1]=(4.0/3.0)*eps_cmb[-1]/(t_cmb[-1])

eps_nu=np.zeros(c.l) #Primordial neutrinos
t_nu=np.zeros(c.l)
t_nu0=(4.0/11.0)**(1.0/3.0)*t_cmb0
t_nu[-1]=t_nu0
eps_nu0=c.om_nu*c.eps_c0
eps_nu[-1]=eps_nu0
s_nu=np.zeros(c.l)
s_nu[-1]=2*np.pi**2*c.kb**4*6*7*t_nu[-1]**3/(45*c.c**3*c.hbar**3*8) #Egan & Lineweaver 2009

eps_l=np.zeros(c.l) #Lambda
eps_l0=c.eps_c0*c.om_l
eps_l[-1]=eps_l0
t_l=np.zeros(c.l)
s_l=np.zeros(c.l)

eps_smbh=np.zeros(c.l) #Supermassive black holes
s_smbh=np.zeros(c.l)
s_smbhhigh=np.zeros(c.l)
s_smbhlow=np.zeros(c.l)

eps_c=np.zeros(c.l) #Critical energy density
eps_c[-1]=c.eps_c0

eps=np.zeros(c.l) #Total energy
eps[-1]=eps_dm0+eps_bm0+eps_cmb0+eps_nu0+eps_l0
s=np.zeros(c.l)