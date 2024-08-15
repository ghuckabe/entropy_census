#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Constants in MKS units, mostly taken from IAU, NIST, Fukugita & Peebles, and Planck

Created on Tue Aug 13 12:07:16 2024

@author: gabbyhuckabee
"""

import numpy as np


#Conversion factors
msol=1.9891e30 #Solar mass, kg, IAU
au=149597870700 #AU, m, IAU
ly=9460730472580800 #Lightyear in m, IAU
mpc=au*10**6/((1/60)*(1/60)*2*np.pi/360) #mpc, m, IAU
t_pl=1.416784e32 #Planck temperature, K, NIST
l_pl=1.616255e-35 #Planck length, m, NIST
m_pl=2.176434e-8 #Plank mass, kg, NIST
rho_pl=m_pl/(l_pl**3) #Planck density, kg/m^3, NIST
ev=1.782661921*10**(-36) #Electronvolt, kg, NIST
yr=365.25*86400  #Year, s, IAU
amu=931.49410242 #Atomic mass unit*c^2, MeV, NIST
ev2K=1.160451812e4 #Electronvolt, K, NIST

#Physical constants 
G=6.67430e-11 #Gravitational constant, N*m^2/kg^2, NIST
c=2.99792458*10**8 #Speed of light, m/s, NIST
kb=1.380649*10**(-23) #Boltzmann constant, J/K, NIST
hbar=1.054571817*10**(-34) #Reduced Planck constant, kg*m^2/s, NIST
sigmat=6.652458*10**(-29) #Thompson scattering cross section, m^2, NIST
sigmasb=5.670374419*10**(-8) #Stefan-Boltzmann constant, W/(m^2*K^4), NIST
me=9.1093837015*10**(-31) #Electron mass, kg, NIST
mp=1.67262192369e-27 #Proton mass, kg, NIST
mn=1.67492749804e-27 #Neutron mass, kg, NIST
mh=1.00794*amu*ev*(10**6) #Hydrogen mass, kg, (PDG, NIST)
e=1.602176634e-19 #Elementary charge, C, NIST

#Cosmological parameters
neff=2.99 #Effective number of neutrino species, Planck, full constraints (3.046 is the Standard model prediction)
k=0. #Curvature of spacetime
h0=67.66#Hubble constant, km/s/Mpc, Planck, p/m 0.42
h0_h=h0+0.42#Hubble constant upper limit
h0_l=h0-0.42 #Hubble constant lower limit
om_bm0=0.02242/(h0/100.0)**2 #Omega baryonic matter present, Planck, p/m 0.00014
om_bm0_h=(0.02242+0.00014)/(h0_l/100.0)**2 #Upper limit Omega baryonic matter present
om_bm0_l=(0.02242-0.00014)/(h0_h/100.0)**2 #Lower limit, Omega baryonic matter present
om_dm0=0.14240/(h0/100.0)**2 - om_bm0 #Omega dark matter present, Planck, p/m 0.00087 first number, 
om_dm0_h=(0.14240+0.00087)/(h0_l/100.0)**2 - om_bm0_h
om_dm0_l=(0.14240-0.00087)/(h0_h/100.0)**2 - om_bm0_l
om_cmb=5.3955157715513285e-5 #Omega CMB, Planck (temp used in eps=ab*t**4)
om_nu=om_cmb*neff*(7./8)*(4/11)**(4./3)
#om_nu=6e-4/(h0/100.0)**2 #Omega neutrinos lower limit
om_l=0.6889 # plmi 0.0056 Omega lambda present Planck
om_l_h=om_l+0.0056
om_l_l=om_l-0.0056
om_remn=0.00048 #Fukugita and Peebles, white dwarves and black holes and neutron stars
om_str0=0.00205#0.0027 #Omega stars, from Fukugita & Peebles
om_k=0 #Omega curvature
eps_c0=c**2*3*(h0*1000/(mpc))**2/(8*np.pi*G) #current critical energy density, mks
eps_c0_h=c**2*3*(h0_h*1000/(mpc))**2/(8*np.pi*G)
eps_c0_l=c**2*3*(h0_l*1000/(mpc))**2/(8*np.pi*G)
bmass=937.12*ev/(10**6) #baryon mass, kg (https://arxiv.org/pdf/astro-ph/0606206.pdf)
eta0=1.38189*10**(18) #mathematica from integrating a from 0 to 1
eta0_h=1.40152*10**(18) #Using lower limits for energy densities and Hubble constant
eta0_l=1.36256*10**(18) #Using upper limints for energy densities and Hubble constant
etainf=5.1733e17+eta0 #Using present time plus Mathematica integral using Lambda-dominated scale factor equation
etainf_h=5.34421*10**(17)+eta0_h #Using lower constant lims
etainf_l=5.00592*10**(17)+eta0_l #Using upper constant lims
t0=4.32752*10**(17) #from mathematica integrating a from 0 to 1 with rad, matter, lambda
t0_h=4.38771*10**(17)
t0_l=4.26826*10**(17)
strdot0=0.013*msol/(60*60*24*365.25*(mpc**3)) #Present time derivative of stellar energy density, Hernquist & Springel 2003
cmbnumdens=4.104*10**8 #m^-3, derived from COBE (Mather 1999)
cmbnumdens_l=(4.104-0.9/100)*10**8
cmbnumdens_h=(4.104+0.9/100)*10**8
photonbaryoneta=6.1e-10#2.75*10**(-8)*om_bm0*(h0/100)**2, WMAP Bennet 2003
photonbaryoneta_l=6.1e-10+0.3e-10
photonbaryoneta_h=6.1e-10-0.2e-10

#Milestones
tdec=2971 #K, decoupled temp when photons and baryons diverge
zdec=1090 #Decoupling redshift for temp above https://people.ast.cam.ac.uk/~pettini/Intro%20Cosmology/Lecture09.pdf
zeq=3380 #matter radiation equality
zrec=1375 #recombination, for X=0.5, half of all baryons ionized

#Early constants
ewphase=100*1e9 #Electroweak phase transition in eV, W and Z bosons become massive and weak interaction cross section changes
qcdphase=150*1e6 #QCD phase transition in eV, quark/gluon interaction becomes important and quarks are bound by gluons in baryons and mesons
nu_dec=0.8*1e6 #Neutrino decoupling energy from Baumann notes (0.8 MeV)
emphase=1 #Electromagnetic phase transition

#Integration parameters
early_length=1000