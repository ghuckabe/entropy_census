#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 16 11:07:03 2020
Entropy Source Census
@author: gabby
"""

import numpy as np
#from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib import rc
import sys, platform, os
import matplotlib
from scipy.special import zeta

rc('text', usetex=True)

#CAMB import
camb_path = '/Users/gabbyhuckabee/Documents/entropy_census-master'
sys.path.append(camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

"""
Constants in MKS
Need better estimate for baryon mass (running average maybe?)
Update DM mass, in GeV currently
Don't know what the baryon degeneracy is (bg)
Don't know what to use for baryon chemical potential (constantly changing?  need approximation for baryon temp)
"""
#Physical constants
G=6.67e-11 #mks
c=3*10**8 #mks
kb=1.38*10**(-23) #Boltzmann constant, mks
hbar=1.055*10**(-34) #reduced planck constant, mks
sigmat=6.65*10**(-29) #Thompson scattering cross section
# sigmasb=5.67*10**(-8) #Stefan-Boltzmann constant
# xe=5*10**(-4) #electron concentration
# xhe=0.081 #Helium concentration
# me=9.109*10**(-31) #electron mass, kg

#Conversion factors
msol=1.989e30 #Solar mass in kg
mpc=3.086e22 #Megaparsec in meters
t_pl=1.41696e32 #K
l_pl=1.616e-33/100 #m
rho_pl=5.15749e96 #Planck density in kg/m^3
m_pl=2.17671e-8 #Plank mass in kg
ly=9.461e15 #Lightyear in m
ev=(5.6096e35)**(-1) #1 eV in kg
yr=365.25*24*3600 #yr to sec

#Cosmological parameters
k=0. #curvature
h0=70. #Hubble constant, km/s/Mpc
om_bm0=0.022/(h0/100.0)**2 #Omega baryonic matter present
om_dm0=0.12/(h0/100.0)**2 #Omega dark matter present
om_cmb=2.47e-5/(h0/100.)**2 #Omega CMB
om_nu=6e-4/(h0/100.0)**2 #Omega neutrinos lower limit
om_l=1-om_bm0-om_dm0-om_cmb-om_nu #Omega lambda present
om_remn=0.00048 #Stolen from Fukugita and Peebles, white dwarves and black holes and neutron stars
om_str0=0.00205#0.0027 #Omega stars, from Egan and Lineweaver
om_k=0 #Omega curvature
eps_c0=c**2*3*(h0*1000/(3.086e22))**2/(8*np.pi*G) #current critical energy density, mks, energy/volume
dmass=40 #dark matter mass, GeV
bmass=1.6*10**(-27) #baryon mass, kg
bg=2 #baryon degeneracy
bmu=1 #baryon chemical potential
eta0=1.48*10**(18) #Present conformal time in seconds (USE CAMB TO FIND THIS)
etainf=9.848743e17 #Conformal time at t=infinity from Egan & Lineweaver
t0=13.799*10**9*yr #Present regular time in seconds (Planck)
#1.7509191904838795e+22  #Conformal time at t=infinity from calculation at end of code
strdot0=0.013*msol/(60*60*24*365.25*(mpc**3)) #Present time derivative of stellar energy density

#Numerical integration parameters
h=3e14#step size in eta, da=sqrt(8*pi*G*eps/3)*a^2*d(eta) from 1.70 Mukhanov
l=4830-251 #array size

"""
Initialize arrays for scale factor, energy density, entropy density, temperature
"""

t=np.zeros(l) #Cosmic time
t[l-1]=t0

eta=np.zeros(l) #Conformal time
eta[l-1]=eta0

a=np.zeros(l) #Scale factor
a[l-1]=1

hub=np.zeros(l) #Hubble constant
hub[l-1]=h0

r_p=np.zeros(l) #Particle Horizon (Observable Universe)
r_p[l-1]=eta0*c

volume=np.zeros(l)
volume[l-1]=4*np.pi*(r_p[l-1])**3/3

r_ceh=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh[l-1]=-c*(etainf-eta0)
s_ceh=np.zeros(l)
s_ceh[l-1]=(r_ceh[l-1]**2*np.pi*kb*c**3/(G*hbar))/volume[l-1]

eps_bm=np.zeros(l) #Baryons
eps_bm0=eps_c0*om_bm0
eps_bm[l-1]=eps_bm0
t_bm=np.zeros(l)
t_bm[l-1]=0.025468041207723818 #This came from crazy old ancient Gabby code from Judd...?
s_bm=np.zeros(l)

eps_str=np.zeros(l) #Stars
eps_str0=om_str0*eps_c0
eps_str[l-1]=eps_str0
s_str=np.zeros(l)

eps_bh=np.zeros(l) #Stellar mass black holes
eps_bh0=0.56*10**8*msol*c**2/mpc**3
eps_bh[l-1]=eps_bh0
s_bh=np.zeros(l)
s_bh0=1.6*10**(17)*kb #Egan and Lineweaver 2009
s_bh[l-1]=s_bh0

eps_dm=np.zeros(l) #Dark matter
eps_dm0=eps_c0*om_dm0
eps_dm[l-1]=eps_dm0
t_dm=np.zeros(l)
s_dm=np.zeros(l)

eps_remn=np.zeros(l) #White dwarves, neutron stars, stellar mass black holes
eps_remn0=eps_c0*om_remn
eps_remn[l-1]=eps_remn0
t_remn=np.zeros(l)
s_remn=np.zeros(l)

#eps_mat=np.zeros(l) #matter
#eps_mat0=eps_dm0+eps_bm0
#eps_mat[l-1]=eps_mat0
#t_mat=np.zeros(l)
##t_mat[l-1]
#s_mat=np.zeros(l)

eps_cmb=np.zeros(l) #Cosmic Microwave Background photons
eps_cmb0=eps_c0*om_cmb
eps_cmb[l-1]=eps_cmb0
t_cmb=np.zeros(l)
t_cmb0=2.73 #K
t_cmb[l-1]=t_cmb0
s_cmb=np.zeros(l)
s_cmb[l-1]=(4.0/3.0)*eps_cmb[l-1]/(t_cmb[l-1])

eps_nu=np.zeros(l) #Primordial neutrinos
t_nu=np.zeros(l)
t_nu0=(4.0/11.0)**(1.0/3.0)*t_cmb0
t_nu[l-1]=t_nu0
eps_nu0=om_nu*eps_c0
eps_nu[l-1]=eps_nu0
s_nu=np.zeros(l)
s_nu[l-1]=2*np.pi**2*kb**4*6*7*t_nu[l-1]**3/(45*c**3*hbar**3*8) #Egan & Lineweaver 2009

eps_l=np.zeros(l) #Lambda
eps_l0=eps_c0*om_l
eps_l[l-1]=eps_l0
t_l=np.zeros(l)
s_l=np.zeros(l)

#eps_rad=np.zeros(l) #Radiation
#t_rad=np.zeros(l)
#t_rad[l-1]=t_nu0+t_cmb0
#s_rad=np.zeros(l)

eps_smbh=np.zeros(l)
s_smbh=np.zeros(l)

eps_c=np.zeros(l) #Critical energy density
eps_c[l-1]=eps_c0

eps=np.zeros(l) #Total energy
eps[l-1]=eps_dm0+eps_bm0+eps_cmb0+eps_nu0+eps_l0
s=np.zeros(l)

arrays=[a,r_p,volume,r_ceh,eps_bm,eps_dm,eps_cmb,eps_nu,eps_l,eps,hub,eps_c,eps_str,t_cmb,t_nu,t_bm,s_smbh,s_bh,s_cmb,s_nu,s_bm,s_ceh]

"""
Functions for a' and a''
"""

def ap(a, eps):
    return np.sqrt(8*np.pi*G*eps*a**4/(3.0*c**2)-k*c**2*a**2)

def ap2(a, epsbm, epsdm, epsl):
    return 4*np.pi*G*(epsdm+epsbm+4*epsl)*a**3/3-k*a

"""
Stellar mass black hole number density calculations, using Chabrier IMF, assume black hole progenitors are greater than 25 Msol,
progenitor to remnant mass function crudely approximated from https://arxiv.org/pdf/astro-ph/9911312.pdf
"""

def strdot(h): #time derivative of stellar mass density, d(rho)/dt
    x=(h/h0)**(2./3.)
    sol=strdot0*x**2/(1+0.012*(x-1)**3*np.exp(0.041*x**(7./4.))) #Hernquist & Springel
    return sol #Peak SFR is around 5-6 z

def prog2rem(m): #progenitor mass to remnant mass, function guessed based on plot in paper above, m in msol
    if m<20:
        sol=1.2
    elif m>42:
        sol=m
    else:
        sol=0.625*m-11.25
    return sol

def imf(m): #initial mass function redshift independent, m in msol
    if m<0.5:
        alpha=-1.35
    else:
        alpha=-2.35
    return m**(alpha+1) #multiply by dlogM to get proportional to dn

def pmf(m): #present main sequence mass function, redshift independent, m in msol
    if m<1:
        sol=imf(m)
    else:
        sol=imf(m)*m**(-2.5)
    return sol #multiply by dlogM to get proportional to dn

def norm(epsstr): #normalization constant between rho_str(z)*1 million yrs (1 star formation event) and imf/pmf integral
    #rstart=10**(-2-((2-(-2))/100))
    #sol=0 #Pre-normalization mtot
    # for r in np.logspace(-2, 2, num=100): #Ranges between 0.01 and 100 Msol
    #     mstep=r-rstart #dM
    #     sol=sol+imf(r)*mstep/(np.log(10)) #Integrates IMF*M over M? most stars are small M
    #     rstart=r #Say sol times A is the total mass in stars that forms in 1 million years (about 1 star formation event time) at some redshift
    sol=27.227
    a=epsstr/(sol*msol*c**2)
    return a

#Stellar mass (in Msol) to lifetime

def tstar(m):
    sol=10**(10)*m**(-2.5)
    return sol

# mstart=10**(1.3-((2-(1.3)/100)))
# def bhnum(m, mstep, epsstr):
# for m in np.logspace(1.3, 2, num=100): #CHANGE THIS TO ONLY INCLUDE BH PROGENITORS
#     mstep=(m-mstart)/(m*np.log(10)) #dlogM
    
#     den=norm(epsstr)*imf(m)*mstep #number density of progenitors between mass M and M+dM
    
#     bhs=bhs+den*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
#     mstart=m

def sbhe(epsstr): #,a): #stellar mass black hole entropy given energy density of stars and scale factor
    bhs=0
    mstart=10**(1.3-((2-(1.3)/100)))
    for m in np.logspace(1.3, 2, num=100): #CHANGE THIS TO ONLY INCLUDE BH PROGENITORS
        mstep=(m-mstart)/(m*np.log(10)) #dM
        
        den=norm(epsstr)*(imf(m)-pmf(m))*mstep #number density of progenitors between mass M and M+dM
        
        bhs=bhs+den*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
        mstep=m
    return bhs
s_bh[l-1]=sbhe(eps_str0)

'''
SMBH functions and DM mass function things from Anthony and Friends
'''

ginf=1.43728 #Does Ginf need a conversion?  Seems dimensionless
xi2=10**(-17) #matter mass per photon (?) squared in 1/msol
rho_l=1.25e-123 #dark energy density in planck density
zeq=2740 #redshift of matter-radiation equality
xeq=om_l*(1+zeq)**(-3)/(om_bm0+om_dm0) #? this is rho_l/rho_m for rho_m at equality, see anthony's paper w max and frank
al=3215 #A_Lambda
q=2e-5 #Scalar fluctuation amplitude on horizon
gconst=0.652 #weak coupling constan at m_z
b=-0.27

def g(zed): #dimensionless
    mass=om_bm0+om_dm0
    x=(om_l/mass)*(1+zed)**(-3)
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
    rho=(epsbm+epsdm)/c**2
    temp=f(m,z,epsbm,epsdm,epsc)*rho/(m*msol)
    om_m=(om_bm0+om_dm0)
    dz=-(1+z)**2
    dx=om_l/om_m*(-3.)*(1+z)**(-4)*dz
    x=om_l/om_m*(1+z)**(-3)
    dgl=1./3.*x**(-2./3.)*(1+(x/ginf**3)**0.795)**(-1./(3.*0.795))*(1-x**0.795/(ginf**(3*0.795)*(1+(x/ginf**3)**0.795)))*dx
    mu=xi2*m
    ds=-du*((9.1*mu**(-2./3.))**b+(50.5*np.log10(834+mu**(-1./3.))-92)**b)**((1-b)/b)*((2./3.)*9.1**b*mu**(-(2*b-3)/3)+50.5*(50.5*np.log10(834+mu**(-1./3.))-92)**(b-1)/(3*mu**(4./3.)*(834+mu**(-1./3.))*np.log(10)))
    dlnsig=-q*(s(m)*(3./2.)*al*dgl+g(z)*ds)/sigma(m, z) #Just needs to be multiplied by da
    return temp*dlnsig #this, times da, will equal dn (number of halos between mass m and m+dm)

'''
Numerical solver for energy density, temperature, and entropy
'''

stop=0
for i in np.arange(2,len(a)+1):
    t[l-i]=t[l-i+1]-h*a[l-i+1]
    
    #Euler Method
    y=a[l-i+1]
    a[l-i]=y-h*ap(y,eps[l-i+1]) #-0.5*h**2*ap2(y,eps_bm[l-i+1],eps_dm[l-i+1],eps_l[l-i+1])/2
    
    #4th order Runge-Kutta attempt
    # k1=h*ap(y,eps[l-i+1]) #h*f(y_n,t_n)
    # k2=h*ap(y+k1/2,(eps[l])) #h*f(y_n+k1/2,t_n+h/2)
    # k3=h*ap(y+k2/2,eps[l]) #h*f(y_n+k2/2,t_n+h/2)
    # k4=h*ap(y+k3, eps[l]) #h*f(y_n+k3,t_n+h)
    # a[l-i]=y+(k1+2*k2+2*k3+k4)/6
    
    if i%100==0:
        print(a[l-i])
#    if a[l-i]<0:
#        stop=l-i+1
#        break
    eta[l-i]=eta0-h*(i-1)    
    r_p[l-i]=a[l-i]*c*(eta0-h*(i-1))
    r_ceh[l-i]=-a[l-i]*(etainf-(eta0-h*(i-1)))*c
    volume[l-i]=4*np.pi*r_ceh[l-i]**3/3
    eps_bm[l-i]=eps_bm[l-i+1]*(a[l-i+1]/a[l-i])**3
    eps_dm[l-i]=eps_dm[l-i+1]*(a[l-i+1]/a[l-i])**3
#    eps_mat[l-i]=eps_dm[l-i]+eps_bm[l-i]
    eps_cmb[l-i]=eps_cmb[l-i+1]*(a[l-i+1]/a[l-i])**4
    eps_nu[l-i]=eps_nu[l-i+1]*(a[l-i+1]/a[l-i])**4
    eps_l[l-i]=eps_l0
#    eps_rad[l-i]=eps_nu[l-i]+eps_cmb[l-i]
    eps[l-i]=eps_bm[l-i]+eps_dm[l-i]+eps_cmb[l-i]+eps_nu[l-i]+eps_l[l-i]
    hub[l-i]=np.sqrt(8*np.pi*G*eps[l-i]/(3*c**2))*mpc/1000 #Assume H=a'/a^2
#    eps_str[l-i]=eps_str[l-i+1]-strdot(hub[l-i+1])*h*a[l-i+1]*c**2
    #comp=32*sigmat*sigmasb*a[l-i+1]*(t_cmb[l-i+1]**4)*xe*(t_bm[l-i+1]-t_cmb[l-i+1])/(3*hub[l-i+1]*me*(c**2)*(1+xe+xhe)*np.sqrt(8*np.pi*G*eps[l-i+1]/3))
    t_cmb[l-i]=((m_pl*c**2*l_pl)**3*eps_cmb[l-i]*15/np.pi**2)**(1./4.)/kb #mukhanov exact for massless relativistic bosons (chemical potential=mass=0)
    t_nu[l-i]=t_cmb[l-i]*(4./11.)**(1./3.) #After electron positron annihilation
    t_bm[l-i]=t_bm[l-i+1]+h*2*(hub[l-i+1]*1000/mpc)*t_bm[l-i+1]*a[l-i+1] #Adiabatic cooling
    eps_c[l-i]=c**2*3*(hub[l-i]*1000/mpc)**2/(8*np.pi*G)
    z=1/a[l-i]-1
    runningtot=0
    iterations=100
    jstart=10**(8.-((17-8)/iterations))
    for j in np.logspace(8, 17, num=iterations): #j (DM mass) is in solar masses
        tempu=xi2*(j-jstart)
        dn=rhs(j, z, eps_bm[l-i], eps_dm[l-i], eps_c[l-i], tempu)*(-h)*ap(a[l-i],eps[l-i]) #should there be a negative in front of the h?  maybe...?
        if dn<0:
            dn=0
        smbh_mass=10**(1.55*np.log10(j/(1e13))+8.01)*msol #smbh mass in kg
        smbh_ds=4*np.pi*kb*G*smbh_mass**2/(c*hbar)
        runningtot=runningtot+smbh_ds*dn #dn is in #/volume already I think
        jstart=j
    s_smbh[l-i]=runningtot
#    s_bh[l-i]=sbhe(eps_str[l-i])
    s_cmb[l-i]=(4.0/3.0)*eps_cmb[l-i]/(t_cmb[l-i]) #From Mukhanov <- natural units, i think this needs to be multiplied by kb to match the lineweaver definition
    s_nu[l-i]=2*np.pi**2*kb**4*6*7*t_nu[l-i]**3/(45*c**3*hbar**3*8) #Lineweaver 2009
#    s_rad[l-i]=s_nu[l-i]+s_cmb[l-i]
    s_bm[l-i]=0
    s_ceh[l-i]=(r_ceh[l-i]**2*np.pi*kb*c**3/(G*hbar))/volume[l-i]

z30=np.argmin(abs(1./a-31))
for zed in np.arange(z30,l):
    eps_str[zed]=eps_str[zed-1]+strdot(hub[zed-1])*h*a[zed-1]*c**2
#    s_bh[zed]=sbhe(eps_str[zed])


mstart=10**(1.3-((2-1.3)/1000))
for i in np.arange(z30,l-1):
    for m in np.logspace(1.3, 2, num=1000):
        mstep=(m-mstart)/(m*np.log(10)) #dlogM
        
        den=norm(eps_str[i])*imf(m)*mstep #dn - number density of progenitors between mass M and M+dM
        intg=a[i]
        steps=1
        while((tstar(m)/h)>intg):
            intg+=a[i+steps]
            steps+=1
        if (i+steps)<l:
            s_bh[i+steps]+=den*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
        mstart=m
        
        
"""
Testing
"""

# for i in np.arange(0,2500):#1685):
#     y=a[l+i-1]
#     a=np.append(a,y+h*ap(y,eps[l+i-1])) #-0.5*h**2*ap2(y,eps_bm[l-i+1],eps_dm[l-i+1],eps_l[l-i+1])/2
#     if i%100==0:
#         print(a[l+i])
#     eta=np.append(eta,eta0+h*i)
#     r_p=np.append(r_p,a[l+i]*c*(eta0+h*i))
#     volume=np.append(volume,4*np.pi*r_p[l+i]**3/3)
#     r_ceh=np.append(r_ceh,-a[l+i]*(etainf-(eta0+h*i))*c)
#     eps_bm=np.append(eps_bm,eps_bm[l+i-1]*(a[l+i-1]/a[l+i])**3)
#     eps_dm=np.append(eps_dm,eps_dm[l+i-1]*(a[l+i-1]/a[l+i])**3)
#     eps_cmb=np.append(eps_cmb,eps_cmb[l+i-1]*(a[l+i-1]/a[l+i])**4)
#     eps_nu=np.append(eps_nu,eps_nu[l+i-1]*(a[l+i-1]/a[l+i])**4)
#     eps_l=np.append(eps_l,eps_l0)
#     eps=np.append(eps,eps_bm[l+i]+eps_dm[l+i]+eps_cmb[l+i]+eps_nu[l+i]+eps_l[l+i])
#     hub=np.append(hub,np.sqrt(8*np.pi*G*eps[l+i]/(3*c**2))*mpc/1000) #Assume H=a'/a^2
#     t_cmb=np.append(t_cmb,((m_pl*c**2*l_pl)**3*eps_cmb[l+i]*15/np.pi**2)**(1./4.)/kb) #mukhanov exact for massless relativistic bosons (chemical potential=mass=0)
#     t_nu=np.append(t_nu,t_cmb[l+i]*(4./11.)**(1./3.)) #After electron positron annihilation
#     eps_c=np.append(eps_c,c**2*3*(hub[l+i]*1000/mpc)**2/(8*np.pi*G))

# runeta=eta0
# detarray=[]

# for p in np.arange(l,l+2000):
#     z=1/a[p]-1
#     if (z<-1 or np.isnan(z)==True):
#         break
#     deta=(1+z)**2*ap(a[p],eps[p])*h/(h0*1000/mpc*np.sqrt(eps[p]))
#     detarray.append(deta)
#     if np.isfinite(deta)==True:
#         runeta+=deta
        

"""
Plots for energy density, temperature, entropy density
"""

#Energy Density    
fig1, ax1 = plt.subplots(1,1)    
ax1.plot(np.log10(a),np.log10(eps_bm), c='b', label='Baryonic Matter')
ax1.plot(np.log10(a),np.log10(eps_cmb), c='r', label="CMB")
ax1.plot(np.log10(a),np.log10(eps_nu), c='y', label='Neutrinos')
ax1.plot(np.log10(a),np.log10(eps_str),c='#ff5733', label="Stars")
#ax1.plot(np.log10(a),np.log10(eps_rad),c='#b15bd6', label='Total Radiation')
ax1.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
ax1.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
#ax1.plot(np.log10(a),np.log10(eps_mat), c='#66ff33', label="Total Matter")
ax1.set_xlabel("log(a)")
ax1.set_ylabel("log(epsilon)")
ax1.set_title("Energy Density vs Scale Factor")
ax1.legend()

#Temperature
fig2, ax2 = plt.subplots(1,1)    
#ax2.plot(np.log10(a),np.log10(tbm), c='b', label='Baryonic Matter')
ax2.plot(np.log10(a),t_cmb, c='r', label="CMB")
ax2.plot(np.log10(a),t_nu, c='y', label='Neutrinos')
#ax2.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax2.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax2.set_xlabel("log(a)")
ax2.set_ylabel("T (K)")
ax2.set_title("Temperature vs Scale Factor")
ax2.legend()

#Entropy Density
fig3, ax3 = plt.subplots(1,1)    
#ax3.plot(np.log10(a),np.log10(s_bm), c='b', label='Baryonic Matter')
ax3.plot(np.log10(a),np.log10(s_cmb), c='r', label=r'$\textrm{CMB}$')
ax3.plot(np.log10(a),np.log10(s_nu), c='#ffe523', label=r'$\nu_{\textrm{primordial}}$')
#ax3.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax3.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax3.plot(np.log10(a),np.log10(s_ceh), c='#c523ff', label=r'$\textrm{CEH}$')
ax3.plot(np.log10(a),np.log10(s_smbh),c='#22ffc6', label=r'$\textrm{Supermassive Black Holes}$')
ax3.plot(np.log10(a),np.log10(s_bh),c='#a8ff23', label=r'$\textrm{Stellar Mass Black Holes}$')
ax3.set_xlabel(r'$\textrm{log}(a)$')
#ax32 = ax3.secondary_xaxis('top', )
ax3.set_ylabel(r'$\textrm{log}(s)$')
ax3.set_title(r'\textrm{Entropy Density vs Scale Factor}')
ax3.set_ylim(bottom=-20)
ax3.legend(loc=2)
fig3.savefig("s_evolution.eps")

#Entropy Total
fig4, ax4 = plt.subplots(1,1)

#ax4.plot(np.log10(a),np.log10(volume*s_bm), c='b', label='Baryonic Matter')
ax4.plot(np.log10(a),np.log10(volume*s_cmb), c='r', label=r'$\textrm{CMB}$')
ax4.plot(np.log10(a),np.log10(volume*s_nu), c='#ffe523', label=r'$\nu_{\textrm{primordial}}$')
#ax4.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax4.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax4.plot(np.log10(a),np.log10(volume*s_ceh), c='#c523ff', label=r'$\textrm{CEH}$')
ax4.plot(np.log10(a),np.log10(volume*s_smbh),c='#22ffc6', label=r'$\textrm{Supermassive Black Holes}$')
ax4.plot(np.log10(a),np.log10(volume*s_bh),c='#a8ff23', label=r'$\textrm{Stellar Mass Black Holes}$')
ax4.set_xlabel(r'$\textrm{log}(a)$')
ax4.set_ylabel(r'$\textrm{log}(S)$')
ax4.set_title(r'\textrm{Entropy vs Scale Factor}')
ax4.set_ylim(bottom=-20)
ax4.legend(loc=2)

#Cosmic Event Horizon
fig4, ax4 = plt.subplots(1,1)    
ax4.plot(np.log10(a),np.log10(r_ceh), label="CEH")
ax4.plot(np.log10(a),np.log10(r_p), label="PH")
ax4.set_xlabel("log(a)")
ax4.set_ylabel("log(R_CEH)")
ax4.legend()
ax4.set_title("Cosmic Event Horizon Radius vs Scale Factor")

print("Present Day Entropy:  S[k] from this code (Egan & Lineweaver 2009)")
print("-------------------")
print("Cosmic Event Horizon       : %10.2E \t(2.6E+122)" % (s_ceh[-2]*volume[-2]/kb))
print("Supermassive Black Holes   : %10.2E \t(1.2E+103)" % (s_smbh[-2]*volume[-2]/kb))
print("Stellar Mass Black Holes   : %10.2E \t(2.2E+96)" % (s_bh[-2]*volume[-2]/kb))
print("Cosmic Microwave Background: %10.2E \t(2.03E+88)" % (s_cmb[-2]*volume[-2]/kb))
print("Primordial Neutrinos       : %10.2E \t(1.93E+88)" % (s_nu[-2]*volume[-2]/kb))