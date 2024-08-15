#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:39:27 2024

@author: gabbyhuckabee
"""

import numpy as np
import constants as c

def npratio(temp): #n_n/n_p in EQ, and n_p~10^(-10)*n_gamma
    massdiff=1.293 #MeV, mass difference between neutron and proton
    return np.exp(-massdiff/temp)

def mue(t): #t in MeV, outputs mu in MeV
    return 10**(0.76*np.log10(t)-8.92)

def mud(t): #t in MeV
    return 10**(4.57*np.tanh(-3.75*(np.log10(t)-1.65))-2.08)

def munu(t): #t in MeV
    if t<170:
        sol=10**(1.31*np.log10(t)-10.03)
    else:
        sol=10**(3.64*np.log10(t)-14.93)
    return sol

def mup(t): #t in eV
    t=t/10**6
    muu=mud(t)+munu(t)-mue(t)
    return 2*muu+mud(t)

def mun(t): #t in eV
    t=t/10**6
    muu=mud(t)+munu(t)-mue(t)
    return muu+2*mud(t)

def gstar(t, pre, post):
    sol=0
    if t>c.qcdphase: #When t is greater than QCD phase transition, quarks are unbound and pions don't exist, and all species are coupled
        for i in pre:
            if i[1]<(t/6.): #if true, then mass is less than temp
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            else: #case when non-relativistic
                pass
    else: #Below the QCD phase transition, use the post-array with bound quarks in pions
        for i in post:
            if i[1]<(t/6.) and i[0]!='Neutrinos': #relativistic
                # if i[0]=='Neutrinos' and t<nu_dec:
                #     sol+=7./8.*i[3]*(t_i/t)**4
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t>c.nu_dec:
                sol+=i[3]*7./8. #Eventually, you can replace with with temperature
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t<c.nu_dec:
                sol+=c.neff*2*7./8.*(4./11.)**(4./3.) #Eventually, you can replace with with temperature
            else: #case when non-relativistic or decoupled neutrinos
                pass
    return sol
            
def gstarS(t, pre, post):
    sol=0
    if t>c.qcdphase: #When t is greater than QCD phase transition, quarks are unbound and pions don't exist, and all species are coupled
        for i in pre:
            if i[1]<(t/6.): #if true, then mass is less than temp
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            else: #case when non-relativistic
                pass
    else: #Below the QCD phase transition, use the post-array with bound quarks in pions
        for i in post:
            if i[1]<(t/6.) and i[0]!='Neutrinos': #relativistic
                # if i[0]=='Neutrinos' and t<nu_dec:
                #     sol+=7./8.*i[3]*(t_i/t)**4
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t>c.nu_dec:
                sol+=i[3]*7./8. #Eventually, you can replace with with temperature
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t<c.nu_dec:
                sol+=c.neff*2*7./8.*(4./11.) #Eventually, you can replace with with temperature
            else: #case when non-relativistic or decoupled neutrinos
                pass
    return sol

def nonrel_energydensity(g, m, T): #in eV for mass and temp
    T=T*c.ev2K
    m=m*c.ev #ev is one electronvolt in kg
    return g*(m*c.c**2/c.l_pl**3)*(m*c.c**2*c.kb*T/(2*np.pi*(c.m_pl*c.c**2)**2))**(3./2.)*np.exp(-m*c.c**2/(c.kb*T))*(1+15*c.kb*T/(8*m*c.c**2))#g*m*(m*T/(2*np.pi))**(3./2.)*np.exp(-m/T)*(1+15*T/(8*m))#*(kb**(3./2.)*c**(1./2.)*G**(-1./2.)*hbar**(-5./2.))

def nonrel_energydensitymu(g, m, T, mu): #in eV temp and mass, and eV for mu
    T=T*c.ev2K #in K now
    m=m*c.ev #in kg now
    mu=mu*c.e # in J now
    if (-m*c.c**2+mu)>0:
        arg=0
    else:
        arg=-m*c.c**2+mu
    return g*(m*c.c**2/c.l_pl**3)*(m*c.c**2*c.kb*T/(2*np.pi*(c.m_pl*c.c**2)**2))**(3./2.)*np.exp((arg)/(c.kb*T))*(1+15*c.kb*T/(8*m*c.c**2))

def sahahyd(T): #in ev, returns n_hydrogen
    muh=mup(T)+mue(T/10**6)
    T=T*c.ev2K
    return 4*(c.mh*T/(c.m_pl*c.t_pl*2*np.pi))**(3./2.)*np.exp((muh*c.ev*10**6-c.mh)/(c.kb*T/c.c**2))/c.l_pl**3
#    return l_pl**3*n**2*(mh*2*np.pi*m_pl**2*c**2/(me*mp*kb*t*ev2K))**(3./2.)*np.exp(13.6/t)
