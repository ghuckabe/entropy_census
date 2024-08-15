"""
Created on Mon Mar 16 11:07:03 2020
Full Version of Entropy Census Code
@author: gabby

When only changing halo mass-SMBH mass relation and DM mass function error (20%)
SMBH lower limit: 2.23E+103
SMBH upper limit: 1.46E+104
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import constants as c
import stellarbh_functions as bh
import smbh_functions as smbh
import early_functions as early
import early_arrays as ea

plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=12)

#Numerical integration parameters
h=1e14#step size in eta, da=sqrt(8*pi*G*eps/3)*a^2*d(eta) from 1.70 Mukhanov
l=100000-86181#4830-224 #array size

#%%
"""
Initialize arrays for scale factor, energy density, entropy density, temperature
"""

time=np.zeros(l) #Cosmic time
time[-1]=c.t0

eta=np.zeros(l) #Conformal time
eta[-1]=c.eta0

a=np.zeros(l) #Scale factor
a[-1]=1
a_ceharg=2982

hub=np.zeros(l) #Hubble constant
hub[-1]=c.h0

r_p=np.zeros(l) #Particle Horizon (Observable Universe)
r_p[-1]=c.eta0*c.c

volume=np.zeros(l) #Volume uses particle horizon
volume[-1]=4*np.pi*(r_p[-1])**3/3

r_ceh=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh[-1]=c.c*(c.etainf-c.eta0)
r_ceh_h=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh_h[-1]=c.c*(c.etainf_h-c.eta0_h)
r_ceh_l=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh_l[-1]=c.c*(c.etainf_l-c.eta0_l)

volumeceh=np.zeros(l) #Volume calculated with CEH
volumeceh[-1]=4*np.pi*(r_ceh[-1])**3/3
volumeceh_h=np.zeros(l)
volumeceh_h[-1]=4*np.pi*(r_ceh_h[-1])**3/3
volumeceh_l=np.zeros(l)
volumeceh_l[-1]=4*np.pi*(r_ceh_l[-1])**3/3

s_ceh=np.zeros(l)
stot_ceh=np.zeros(l)
s_ceh[-1]=(r_ceh[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh[-1]
stot_ceh[-1]=(r_ceh[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
s_ceh_h=np.zeros(l)
stot_ceh_h=np.zeros(l)
s_ceh_h[-1]=(r_ceh_h[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_h[-1]
stot_ceh_h[-1]=(r_ceh_h[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
s_ceh_l=np.zeros(l)
stot_ceh_l=np.zeros(l)
s_ceh_l[-1]=(r_ceh_l[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_l[-1]
stot_ceh_l[-1]=(r_ceh_l[-1]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))

eps_bm=np.zeros(l) #Baryons
eps_bm0=c.eps_c0*c.om_bm0
eps_bm[-1]=eps_bm0
t_bm=np.zeros(l)
t_bm[-1]=0.025468041207723818 #This came from old ASU code from Judd Bowman?
s_bm=np.zeros(l)

#Eagle data
bm_eagleresults=[148.4053081304242, 84.3039732671589, 418.17238398806165, 853.2427715295006, 1572.864365719125, 1810.0005155900126, 1759.1301851009557, 2699.516883529645, 2110.990845022963, 2226.961377352639, 3126.284474107758, 4430.343761110442, 6127.405729410453, 4862.621531291029, 7113.963501171243, 8153.852287086782, 6080.247274201631, 8615.433172615767, 1927.9765390416364, 8000.553030185015, 1305.644296343521, 2968.6629285091212, 824.4928913315442, 1269.6081292868707, 464.85246944704045, 280.57777072388836, 233.8795437672676, 232.7706791309389, 235.5692882942148]
bmz=np.array([0.00, 0.10, 0.18, 0.27, 0.37, 0.50, 0.62, 0.74, 0.87, 1.00, 1.26, 1.49, 1.74, 2.01, 2.24, 2.48, 3.02, 3.53, 3.98, 4.49, 5.04, 5.49, 5.97, 7.05, 8.07, 8.99, 9.99, 15.13, 20.00])
bma=1/(bmz+1)
s_bm[-1]=bm_eagleresults[0]*c.kb

eps_str=np.zeros(l) #Stars
eps_str0=c.om_str0*c.eps_c0
eps_str[-1]=eps_str0
s_str=np.zeros(l)

eps_bh=np.zeros(l) #Stellar mass black holes
eps_bh0=0.56*10**8*c.msol*c.c**2/c.mpc**3 #Fukugita & Peebles
eps_bh[-1]=eps_bh0
s_bh=np.zeros(l)
s_bh0=1.6*10**(17)*c.kb #Egan and Lineweaver 2009
s_bh[-1]=s_bh0
s_bhhigh=np.zeros(l)
s_bhlow=np.zeros(l)
s_bh_err=np.zeros(l)
s_bh_sicilia=np.zeros(l)

eps_dm=np.zeros(l) #Dark matter
eps_dm0=c.eps_c0*c.om_dm0
eps_dm[-1]=eps_dm0
t_dm=np.zeros(l)
s_dm=np.zeros(l)

eps_remn=np.zeros(l) #White dwarves, neutron stars, stellar mass black holes
eps_remn0=c.eps_c0*c.om_remn
eps_remn[-1]=eps_remn0
t_remn=np.zeros(l)
s_remn=np.zeros(l)

eps_cmb=np.zeros(l) #Cosmic Microwave Background photons
eps_cmb0=c.eps_c0*c.om_cmb
eps_cmb[-1]=eps_cmb0
t_cmb=np.zeros(l)
t_cmb0=2.7255 #K, Fixen 2009
t_cmb[-1]=t_cmb0
s_cmb=np.zeros(l)
s_cmb[-1]=(4.0/3.0)*eps_cmb[-1]/(t_cmb[-1])

eps_nu=np.zeros(l) #Primordial neutrinos
t_nu=np.zeros(l)
t_nu0=(4.0/11.0)**(1.0/3.0)*t_cmb0
t_nu[-1]=t_nu0
eps_nu0=c.om_nu*c.eps_c0
eps_nu[-1]=eps_nu0
s_nu=np.zeros(l)
s_nu[-1]=2*np.pi**2*c.kb**4*6*7*t_nu[-1]**3/(45*c.c**3*c.hbar**3*8) #Egan & Lineweaver 2009

eps_l=np.zeros(l) #Lambda
eps_l0=c.eps_c0*c.om_l
eps_l[-1]=eps_l0
t_l=np.zeros(l)
s_l=np.zeros(l)

eps_smbh=np.zeros(l) #Supermassive black holes
s_smbh=np.zeros(l)
s_smbhhigh=np.zeros(l)
s_smbhlow=np.zeros(l)

eps_c=np.zeros(l) #Critical energy density
eps_c[-1]=c.eps_c0

eps=np.zeros(l) #Total energy
eps[-1]=eps_dm0+eps_bm0+eps_cmb0+eps_nu0+eps_l0
s=np.zeros(l)
#%%
"""
Functions for a' and a'', conformal time derivatives
"""

def ap(a, epst):
    return np.sqrt(8*np.pi*c.G*epst*a**4/(3.0*c.c**2)-c.k*c.c**2*a**2)

def ap2(a, epsbm, epsdm, epsl):
    return 4*np.pi*c.G*(epsdm+epsbm+4*epsl)*a**3/(3*c.c**2)-c.k*a

"""
Stellar mass black hole number density calculations
Uses Chabrier IMF, assume black hole progenitors are greater than 25 Msol
Progenitor to remnant mass function crudely approximated from Fryer & Kalogera, 2001
https://arxiv.org/pdf/astro-ph/9911312.pdf
"""

mtest=np.logspace(0.8, 2.3, num=1000)
yz0=bh.dn_sicilia(mtest, 5)
figt, axt = plt.subplots(1,1)
axt.scatter(np.log10(mtest), np.log10(yz0))
#axt.set_ylim(-3,7)
axt.set_xlim(0.8, 2.3)

#%%
#Present day SMBH entropy
runningtot=0
runningtothigh=0
runningtotlow=0
iterations=100
jstart=10**(8.-((17-8)/iterations))
for j in np.logspace(8, 17, num=iterations): #j (DM mass) is in solar masses
    tempu=smbh.xi2*(j-jstart)
    dn=smbh.rhs(j, 0, eps_bm[-1], eps_dm[-1], eps_c[-1], tempu)*(-h)*ap(1,eps[-1]) #should there be a negative in front of the h?  maybe...?
    if dn<0:
        dn=0
    dnhigh=dn*1.2
    dnlow=dn*0.8
    smbh_mass=10**(1.55*np.log10(j/(1e13))+8.01)*c.msol #smbh mass in kg
    smbh_masshigh=10**(1.6*np.log10(j/(1e13))+8.05)*c.msol
    smbh_masslow=10**(1.5*np.log10(j/(1e13))+7.97)*c.msol
    smbh_dshigh=4*np.pi*c.kb*c.G*smbh_masshigh**2/(c.c*c.hbar)
    smbh_dslow=4*np.pi*c.kb*c.G*smbh_masslow**2/(c.c*c.hbar)
    smbh_ds=4*np.pi*c.kb*c.G*smbh_mass**2/(c.c*c.hbar)
    runningtot=runningtot+smbh_ds*dn #dn is in #/volume already I think
    runningtothigh=runningtothigh+smbh_dshigh*dnhigh
    runningtotlow=runningtotlow+smbh_dslow*dnlow
    jstart=j
    
s_smbh[-1]=runningtot
s_smbhhigh[-1]=runningtothigh
s_smbhlow[-1]=runningtotlow
#%%
'''
Baryon entropy density functions
Species tracking arrays go as follows:
[electrons, protons, singly ionized helium, neutral hydrogen, neutral helium, heavy metals]
'''

specpart=[2, 2, 2, 4, 1]
specmass=[0.5109989461, 938.272081, 3.01603*c.amu, 2.014*c.amu, 4.00260*c.amu] #MeV (PDG, NIST)
ionization_energy=np.array([13.54, 24.48, 54.17])*1.60218*10**(-19)
#%%

'''
Numerical solver for energy density, temperature, and entropy
'''
s_smbhmf=[]
decoupling_index=0

print("Scale factor:")

stop=0
switch=False
for i in np.arange(2,len(a)+1):
    #Euler Method
    y=a[-i+1]
    a[-i]=y-h*ap(y,eps[-i+1])#-0.5*h**2*ap2(y,eps_bm[l-i+1],eps_dm[l-i+1],eps_l[l-i+1])/2
    
    #4th order Runge-Kutta attempt
    # k1=h*ap(y,eps[l-i+1]) #h*f(y_n,t_n)
    # k2=h*ap(y+k1/2,(eps[l])) #h*f(y_n+k1/2,t_n+h/2)
    # k3=h*ap(y+k2/2,eps[l]) #h*f(y_n+k2/2,t_n+h/2)
    # k4=h*ap(y+k3, eps[l]) #h*f(y_n+k3,t_n+h)
    # a[l-i]=y+(k1+2*k2+2*k3+k4)/6
    
    if i%100==0:
        print(a[-i])
#    if a[l-i]<0:
#        stop=l-i+1
#        break
    eta[-i]=c.eta0-h*(i-1)    
    r_p[-i]=a[-i]*c.c*(c.eta0-h*(i-1))
    r_ceh[-i]=a[-i]*(c.etainf-(c.eta0-h*(i-1)))*c.c
    r_ceh_h[-i]=a[-i]*(c.etainf_h-(c.eta0_h-h*(i-1)))*c.c
    r_ceh_l[-i]=a[-i]*(c.etainf_l-(c.eta0_l-h*(i-1)))*c.c
    volumeceh[-i]=4*np.pi*r_ceh[-i]**3/3
    volumeceh_h[-i]=4*np.pi*r_ceh_h[-i]**3/3
    volumeceh_l[-i]=4*np.pi*r_ceh_l[-i]**3/3
    volume[-i]=4*np.pi*r_p[-i]**3/3
    eps_bm[-i]=eps_bm[-i+1]*(a[-i+1]/a[-i])**3
    eps_dm[-i]=eps_dm[-i+1]*(a[-i+1]/a[-i])**3
#    eps_mat[-i]=eps_dm[-i]+eps_bm[-i]
    eps_cmb[-i]=eps_cmb[-i+1]*(a[-i+1]/a[-i])**4
    eps_nu[-i]=eps_nu[-i+1]*(a[-i+1]/a[-i])**4
    eps_l[-i]=eps_l0
#    eps_rad[-i]=eps_nu[-i]+eps_cmb[-i]
    eps[-i]=eps_dm[-i]+eps_cmb[-i]+eps_nu[-i]+eps_l[-i]+eps_bm[-i]
    hub[-i]=np.sqrt(8*np.pi*c.G*eps[-i]/(3*c.c**2))*c.mpc/1000
    if (eps_bm[-i+1]+eps_dm[-i+1])>eps_l[-i+1]:
        time[-i]=time[-i+1]*(a[-i]/a[-i+1])**(3./2.)
    else:
        time[-i]=(np.log(a[-i]/a[-i+1])+hub[-i+1]*time[-i+1])/hub[-i] #Assume H=a'/a^2
#    eps_str[-i]=eps_str[-i+1]-strdot(hub[-i+1])*h*a[-i+1]*c.c**2
    #comp=32*sigmat*sigmasb*a[l-i+1]*(t_cmb[l-i+1]**4)*xe*(t_bm[l-i+1]-t_cmb[l-i+1])/(3*hub[l-i+1]*me*(c**2)*(1+xe+xhe)*np.sqrt(8*np.pi*G*eps[l-i+1]/3))
    t_cmb[-i]=((c.m_pl*c.c**2*c.l_pl)**3*eps_cmb[-i]*15/(np.pi**2))**(1./4.)/c.kb #mukhanov exact for massless relativistic bosons (chemical potential=mass=0)
    if t_cmb[-i]/c.ev2K>0.27 and switch==False:
        decoupling_index=l-i
        switch=True
    else:
        pass
    t_nu[-i]=t_cmb[-i]*(4./11.)**(1./3.) #After electron positron annihilation
    #t_bm[l-i]=t_bm[l-i+1]+h*2*(hub[l-i+1]*1000/mpc)*t_bm[l-i+1]*a[l-i+1] #Adiabatic cooling
    eps_c[-i]=c.c**2*3*(hub[-i]*1000/c.mpc)**2/(8*np.pi*c.G)
    z=1/a[-i]-1
    # if z>20 and switch==False:
    #     s_bm[l-i:]=np.interp(a[l-i:], np.flip(bma), np.flip(bm_eagleresults))*kb
    #     #s_bm[l-i]=bm_eagleresults[-1]*kb
    #     switch==True
    # elif z>20 and switch==True:
    #     s_bm[l-i]=eps_bm[l-i]/t_bm[l-i]#s_bm[l-i+1]-(eps_bm[l-i+1]-eps_bm[l-i])/t_bm[l-i]
    # else:
    #     pass
    runningtot=0
    runningtothigh=0
    runningtotlow=0
    iterations=100
    jstart=10**(8.-((17-8)/iterations))
    smftemp=[]
    for j in np.logspace(8, 17, num=iterations): #j (DM mass) is in solar masses
        tempu=smbh.xi2*(j-jstart)
        dn=smbh.rhs(j, z, eps_bm[-i], eps_dm[-i], eps_c[-i], tempu)*(-h)*ap(a[-i],eps[-i]) #should there be a negative in front of the h?  maybe...?
        if dn<0:
            dn=0
        dnhigh=dn*1.2
        dnlow=dn*0.8
        smbh_mass=10**(1.55*np.log10(j/(1e13))+8.01)*c.msol #smbh mass in kg
        smbh_masshigh=10**(1.6*np.log10(j/(1e13))+8.05)*c.msol
        smbh_masslow=10**(1.5*np.log10(j/(1e13))+7.97)*c.msol
        smbh_dshigh=4*np.pi*c.kb*c.G*smbh_masshigh**2/(c.c*c.hbar)
        smbh_dslow=4*np.pi*c.kb*c.G*smbh_masslow**2/(c.c*c.hbar)
        smbh_ds=4*np.pi*c.kb*c.G*smbh_mass**2/(c.c*c.hbar)
        runningtot=runningtot+smbh_ds*dn #dn is in #/volume already I think
        runningtothigh=runningtothigh+smbh_dshigh*dnhigh
        runningtotlow=runningtotlow+smbh_dslow*dnlow
        smftemp.append(smbh_ds*dn)
        jstart=j
    s_smbhmf.append(smftemp)
        
    s_smbh[-i]=runningtot
    s_smbhhigh[-i]=runningtothigh
    s_smbhlow[-i]=runningtotlow
    s_cmb[-i]=2*np.pi**2*c.kb**4*2*t_cmb[-i]**3/(45*c.c**3*c.hbar**3)
    s_nu[-i]=2*np.pi**2*c.kb**4*6*7*t_nu[-i]**3/(45*c.c**3*c.hbar**3*8) #Lineweaver 2009
#    s_rad[l-i]=s_nu[l-i]+s_cmb[l-i]
    stot_ceh[-i]=(r_ceh[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
    s_ceh[-i]=(r_ceh[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh[-i]
    stot_ceh_h[-i]=(r_ceh_h[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
    s_ceh_h[-i]=(r_ceh_h[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_h[-i]
    stot_ceh_l[-i]=(r_ceh_l[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))
    s_ceh_l[-i]=(r_ceh_l[-i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/volumeceh_l[-i]
#%%
print("\n Now calculating stellar energy density, stellar mass black holes, and baryons... \n")
#Stellar energy density plotting
z20=np.argmin(abs(1./a-21))
z30=np.argmin(abs(1./a-31))
for zed in np.arange(z30,l):
    eps_str[zed]=eps_str[zed-1]+bh.strdot(hub[zed-1])*h*a[zed-1]*c.c**2
    print(bh.strdot(hub[zed-1])*h*a[zed-1]*c.c**2)

dentestarray=np.zeros(l)
densiciliatestarray=np.zeros(l)

#Stellar mass black hole number density tracking, typically at 8.05E+97
mstart=10**(1.3-((2-1.3)/1000))
mstart=10**(1.398-((2-1.3)/1000)) #start at M=25 Msol, s=8.05e+97 mstart has negligible change, most comes from high mass BHs
for i in np.arange(z30,l-1):
    for m in np.logspace(1.398, 2, num=1000):
        mstep=(m-mstart)/(m*np.log(10)) #dlogM
        epsstr_step=bh.strdot(hub[i])*h*a[i]*c.c**2
        den, den_err=bh.norm(epsstr_step)*bh.imf(m)[0]*mstep, bh.norm(epsstr_step)*bh.imf(m)[1]*mstep #dn, dn_err - number density of progenitors between mass M and M+dM
        #den, den_err=norm(eps_str[i])*imf(m)[0]*mstep, norm(eps_str[i])*imf(m)[1]*mstep #dn, dn_err - number density of progenitors between mass M and M+dM
        denhigh=bh.normhigh(epsstr_step)*bh.imfhigh(m)*mstep
        denlow=bh.normlow(epsstr_step)*bh.imflow(m)*mstep
        z=1/a[i]-1
        if z<=10:
            den_sicilia=bh.dn_sicilia(m,z)*mstep
            densiciliatestarray[i]=den_sicilia
        #a*deta=dt
        #t=int from a[i] to a[i+steps] of a*deta
        intg=a[i]*h
        steps=1
        while(bh.tstar(m)>intg):
            intg+=a[i+steps]*h
            steps+=1
        if (i+steps)<l:
            s_bh[i+steps:]+=den*(4*np.pi*c.kb*c.G/(c.c*c.hbar))*(bh.prog2rem(m)*c.msol)**2
            if z<=10:
                s_bh_sicilia[i+steps]+=den_sicilia*(4*np.pi*c.kb*c.G/(c.c*c.hbar))*(bh.prog2rem(m)*c.msol)**2
 #           s_bh_err[i+steps]+=np.sqrt((4*np.pi*k_b*G/(c*hbar))**2*(den_err**2*(prog2rem(m)*msol)**2+()))
            s_bhhigh[i+steps]+=denhigh*(4*np.pi*c.kb*c.G/(c.c*c.hbar))*(bh.prog2rem(m)*c.msol)**2
            s_bhlow[i+steps]+=denlow*(4*np.pi*c.kb*c.G/(c.c*c.hbar))*(bh.prog2rem(m)*c.msol)**2
        mstart=m

s_bh_sicilia=np.array(s_bh_sicilia)

mf=[]

for zed in np.linspace(0,10,num=50):
    mftemp=[]
    mstart=10**(0.69897-((2.20412-0.69897)/1000)) #start atm=5 to m=160
    for m in np.logspace(0.69897, 2.20412, num=1000):
        mstep=(m-mstart)/(m*np.log(10)) #dlogM
        mftemp.append(bh.dn_sicilia(m, zed)*mstep*(4*np.pi*c.kb*c.G/(c.c*c.hbar))*(m*c.msol)**2)
        mstart=m
    mf.append(mftemp)

marray=np.logspace(0.69897, 2.20412, num=1000)
peaksbdensity=[]
for i in mf:
    massarg=np.argmax(i)
    peaksbdensity.append(np.log10(marray[massarg]))
bhfg, bhax = plt.subplots(1,1)
bhax.scatter(np.linspace(0,10,num=50), peaksbdensity)
bhax.set_xlabel("Redshift")
bhax.set_ylabel(r"$M_{BH}$")
bhax.set_title(r"Mass Bin with Highest $s_{BH}$")


#%%
#Baryons
z20=np.argmin(abs(1./a-21))
s_bm[z20:]=np.interp(a[z20:], np.flip(bma), np.flip(bm_eagleresults))*c.kb
#s_bm[l-i]=bm_eagleresults[-1]*kb
deg=2
n_b=c.photonbaryoneta*c.cmbnumdens/a**3 #Baryon number density in m^-3

# https://arxiv.org/pdf/astro-ph/9909275.pdf
# https://arxiv.org/pdf/astro-ph/9912182.pdf
ar=4*c.sigmasb/c.c
def dtm(tr, tm, ne, nb, hubc, a):
    lambdaterms=0
    sol=8*c.sigmat*ar*tr**4*ne*(tm-tr)/(3*hubc*c.me*c.c*nb)+2*tm+2*lambdaterms/(3*c.kb*nb*h)
    return sol*a #dT/dz

def Xfrac(t): #https://people.ast.cam.ac.uk/~pettini/Intro%20Cosmology/Lecture09.pdf
    q=ionization_energy[0]
    s=3.84*c.photonbaryoneta*(c.kb*t/(c.me*c.c**2))**1.5*np.exp(q/(c.kb*t))
    return 1/(1+s)

adecarg=np.argmin(abs(a-1/(1+c.zdec)))
t_bm[:(adecarg+1)]=t_cmb[:(adecarg+1)]
hmm=np.zeros(len(t_bm))

nearray=[]
for i in np.arange(adecarg, len(t_bm)-1):
    ne=n_b[i]
    #ne=Xfrac(t_bm[i])*n_b[i]
    nearray.append(ne)
    hmm[i]=dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i], a[i])
    #t_bm[i+1]=t_bm[i]-hub[i]*dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i])*h*1000/mpc
    t_bm[i+1]=t_bm[i]-a[i]**(-2)*dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i], a[i])*(a[i+1]-a[i])

# adecarg=np.argmin(abs(a-1/(1+zdec)))
# t_bm[:(adecarg+1)]=t_cmb[:(adecarg+1)]
# hmm=np.zeros(len(t_bm))

# nearray=[]
# for i in np.arange(adecarg, len(t_bm)-1):
#     ne=n_b[i]
#     #ne=Xfrac(t_bm[i])*n_b[i]
#     nearray.append(ne)
#     hmm[i]=dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i], a[i])
#     #t_bm[i+1]=t_bm[i]-hub[i]*dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i])*h*1000/mpc
#     t_bm[i+1]=t_bm[i]-a[i]**(-2)*dtm(t_cmb[i], t_bm[i], ne, n_b[i], hub[i], a[i])*(a[i+1]-a[i])
    
#s_bm[:z20]=n_b[:z20]*(5./2.-np.log((hbar**3/kb**(3./2.))*n_b[:z20]*(2*np.pi/(t_bm[:z20]*bmass))**(3./2.)/(deg*(1+15.*t_bm[:z20]*kb/(8.*(bmass*c**2))))))*kb

s_bm=n_b*(5./2.-np.log((c.hbar**3/c.kb**(3./2.))*n_b*(2*np.pi/(t_bm*c.bmass))**(3./2.)/(deg*(1+15.*t_bm*c.kb/(8.*(c.bmass*c.c**2))))))*c.kb

s_bm[z20:]=np.interp(a[z20:], np.flip(bma), np.flip(bm_eagleresults))*c.kb
#%%

'''
Early Universe
We approximate interaction rates from dimensional analysis of the cross section and temperature dependence of the number density, ignoring v (Gamma=sigma*n*v)
'''
print("\n Now calculating pre-CMB era... \n")


earlytemp=np.logspace(np.log10(t_cmb[decoupling_index]/c.ev2K),11, c.early_length) #eV

ea.early_dm[-1]=eps_dm[decoupling_index]
ea.early_l[-1]=eps_l[decoupling_index]

e_dec=ea.standardmodel_postqcd[11][1]

earlytemp=np.flip(earlytemp)
eps_rel=np.zeros(len(earlytemp))
s_rel=np.zeros(len(earlytemp))
early_a=np.zeros(len(earlytemp))
early_a[-1]=a[decoupling_index]*(t_cmb[decoupling_index]/c.ev2K)*early.gstarS(t_cmb[decoupling_index]/c.ev2K, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(earlytemp[-1]*early.gstarS(earlytemp[-1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
early_hub=np.zeros(len(earlytemp))
early_hub[-1]=hub[decoupling_index]

# #%%
# mutest=np.logspace(-10, np.log10(mup(0.0000001)), 100)
# ttest=np.logspace(np.log10(0.01), np.log10(qcdphase), 100)
# data=[]



# for j in np.arange(0,len(ttest)):
#     data.append([])
#     for k in np.arange(0, len(mutest)):
#         data[j].append(nonrel_energydensitymu(4, mp/ev, ttest[j], mutest[k]*10**6))

# figtest, axtest = plt.subplots()
# CS = axtest.contourf(mutest,ttest,data, c=np.log10(data))
# cbar=figtest.colorbar(CS)
# cbar.ax.set_ylabel("Energy Density")

#%%

#For T=0.1 MeV, neutrons and protons exist, but not combined into nuclei
#0.3 eV is recombination
#T=0.27 eV is decoupling, when CMB is released
backwardsindex_array=np.flip(np.arange(0, len(earlytemp)))

backwardsindex_fromdecoupling=np.flip(np.arange(0, c.early_length))

a_qcd=0
a_nudec=0 #Scale factor at neutrino decoupling (set by for loop)
a_edec=0 #Scale factor at electron positron annihilation (set by for loop)

#CHECK

for i in backwardsindex_fromdecoupling:
    t=earlytemp[i]
    if i!=(c.early_length-1):
        #print('test')
        early_a[i]=early_a[i+1]*earlytemp[i+1]*early.gstarS(earlytemp[i+1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(t*early.gstarS(t, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
        ea.early_dm[i]=ea.early_dm[i+1]*(early_a[i+1]/early_a[i])**3
        ea.early_l[i]=ea.early_l[i+1]
    else:
        pass

#DM Freeze Out (for M=1 GeV WIMPs, freeze out is 100 MeV, aka less by a factor of 10)
mev100=np.argmin(abs(earlytemp-100*10**6))
ea.early_dm[:mev100]=np.zeros(mev100)

backwardsindex_array=np.arange(0, len(earlytemp))

backwardsindex_fromdecoupling=np.arange(0, c.early_length)

testswitch1=0

testswitch2=0

for i in backwardsindex_fromdecoupling:
    t=earlytemp[i]
    #print(t/10**6)
    eps_rel[i]=np.pi**2*early.gstar(t, ea.standardmodel_preqcd, ea.standardmodel_postqcd)*(c.kb*t*c.ev2K)**4/(30*(c.m_pl*c.c**2*c.l_pl)**3)
    s_rel[i]=(2*np.pi**2*early.gstarS(t, ea.standardmodel_preqcd, ea.standardmodel_postqcd)*(t*c.ev2K)**3/45)*c.kb**4/(c.c**3*c.hbar**3)
    if t>c.qcdphase: #Pre QCD phase transition, early universe
        for particle in ea.standardmodel_preqcd:
            if (t/6.)<particle[1]: #then particle is non-rel/decoupled, has its own array
                af=early_a[-1]*earlytemp[-1]*early.gstarS(earlytemp[-1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(particle[1]*6*early.gstarS(particle[1]*6, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=early.nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
    elif t<c.qcdphase and t>c.nu_dec: #Post confinement, pre weak decoupling
        if a_qcd==0:
            a_qcd=early_a[i]
        else:
            pass
        for particle in ea.standardmodel_postqcd:
            if (t/6.)<particle[1] and particle[0]!='Neutrinos' and particle[0]!='Electron': #then particle has its own array
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*early.gstarS(earlytemp[decoupling_index-1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(particle[1]*6*early.gstarS(particle[1]*6, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=early.nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Electron':
                particle_t=t
                energy=7*np.pi**2*particle[3]*(particle_t*c.kb*c.ev2K)**4/(120*(c.m_pl*c.c**2*c.l_pl)**3)*(1+30*(early.mue(particle_t/10**6)*10**6/particle_t)**2/(7*np.pi**2))
                entropy=0
                #entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
        ea.early_proton[i]=early.nonrel_energydensitymu(4, c.mp/c.ev, t, early.mup(t)*10**6) #Check degeneracy later
        ea.early_neutron[i]=early.nonrel_energydensitymu(4, c.mn/c.ev, t, early.mun(t)*10**6)
        ea.early_hydrogen[i]=c.mh*early.sahahyd(t)
        ea.early_mup[i]=early.mup(t)
        ea.early_mun[i]=early.mun(t)
        ea.protontemp[i]=t
        ea.neutrontemp[i]=t
        ea.early_s_proton[i]=ea.early_proton[i]/ea.protontemp[i]
        ea.early_s_neutron[i]=ea.early_neutron[i]/ea.neutrontemp[i]
    elif t<c.nu_dec and t>e_dec:
        if a_nudec==0:
            a_nudec=early_a[i]
            protaf=early_a[i]
            protfreeze=t
        else:
            pass
        for particle in ea.standardmodel_postqcd:
            if (t/6.)<particle[1] and particle[0]!='Neutrinos' and particle[0]!='Electron': #then nonrelativistic particle has its own array and isn't decoupled neutrinos
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*early.gstarS(earlytemp[decoupling_index-1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(particle[1]*6*early.gstarS(particle[1]*6, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=early.nonrel_energydensity(particle[3], particle[1], particle_t)
                entrop=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Neutrinos':
                particle_t=c.nu_dec*(a_nudec/early_a[i])
                energy=7*np.pi**2*particle[3]*(particle_t*c.kb*c.ev2K)**4/(120*(c.m_pl*c.c**2*c.l_pl)**3)
                entropy=2*np.pi**2*c.kb**4*6*7*(particle_t*c.ev2K)**3/(45*c.c**3*c.hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
                # print("Calculating a neutrino point")
                # print(particle_t)
                # print(i)
            elif particle[0]=='Electron':
                particle_t=t
                energy=7*np.pi**2*particle[3]*(particle_t*c.kb*c.ev2K)**4/(120*(c.m_pl*c.c**2*c.l_pl)**3)*(1+30*(early.mue(particle_t/10**6)*10**6/particle_t)**2/(7*np.pi**2))
                entropy=0
                #entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
        ea.protontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        ea.neutrontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        ea.early_proton[i]=early.nonrel_energydensitymu(4, c.mp/c.ev, ea.protontemp[i], early.mup(ea.protontemp[i])*10**6) #Check degeneracy later
        ea.early_neutron[i]=early.nonrel_energydensitymu(4, c.mn/c.ev, ea.neutrontemp[i], early.mun(ea.neutrontemp[i])*10**6)
        ea.early_hydrogen[i]=early.sahahyd(t)*c.mh
        ea.early_s_proton[i]=ea.early_proton[i]/ea.protontemp[i]
        ea.early_s_neutron[i]=ea.early_neutron[i]/ea.neutrontemp[i]
    # elif t<e_dec and t>emphase:
        
    else:
        if a_edec==0:
            a_edec=early_a[i]
        else:
            pass
        for particle in ea.standardmodel_postqcd:
            if particle[0]=='Electron': #then nonrelativistic particle has its own array and isn't decoupled neutrinos
                particle_t=t#e_dec*(a_edec/early_a[i])**2
                energy=early.nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Photon':
                #particle_t=nu_dec*(a_nudec/early_a[i])*(11./4)**(1./3)
                particle_t=(t_cmb[0]/c.ev2K)*(a[0]/early_a[i])
                energy=np.pi**2*particle[3]*(c.kb*c.ev2K*particle_t)**4/(30*(c.m_pl*c.c**2*c.l_pl)**3)
                #np.pi**2*gstar(t)*(kb*t*ev2K)**4/(30*(m_pl*c**2*l_pl)**3)
                entropy=2*np.pi**2*c.kb**4*2*(particle_t*c.ev2K)**3/(45*c.c**3*c.hbar**3)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Neutrinos':
                particle_t=c.nu_dec*(a_nudec/early_a[i])
                energy=7*np.pi**2*particle[3]*(particle_t*c.kb*c.ev2K)**4/(120*(c.m_pl*c.c**2*c.l_pl)**3)
                entropy=2*np.pi**2*c.kb**4*6*7*(particle_t*c.ev2K)**3/(45*c.c**3*c.hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
                # particle_t=particle[1]*6*(af/early_a[i])
                # energy=7*np.pi**2*particle[3]*(kb*ev2K*particle_t)**4/(120*(m_pl*c**2*l_pl)**3)
                # particle[4][i]=energy
            else: #particle is nonrel and not a photon, neutrino, or electron
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*early.gstarS(earlytemp[decoupling_index-1], ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.)/(particle[1]*6*early.gstarS(particle[1]*6, ea.standardmodel_preqcd, ea.standardmodel_postqcd)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=early.nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
        ea.protontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        ea.neutrontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        ea.early_proton[i]=early.nonrel_energydensitymu(4, c.mp/c.ev, ea.protontemp[i], early.mup(ea.protontemp[i])*10**6) #Check degeneracy later
        ea.early_neutron[i]=early.nonrel_energydensitymu(4, c.mn/c.ev, ea.neutrontemp[i], early.mun(ea.neutrontemp[i])*10**6)
        ea.early_hydrogen[i]=early.sahahyd(t)*c.mh
        ea.early_s_proton[i]=ea.early_proton[i]/ea.protontemp[i]
        ea.early_s_neutron[i]=ea.early_neutron[i]/ea.neutrontemp[i]
        

#For hydrogen reionization, use analytic formulas from Mukhanov or other textbooks

#Energy Density
total_bm=ea.standardmodel_postqcd[7][4][0]+ea.standardmodel_postqcd[8][4][0]
for i in range(len(ea.standardmodel_postqcd)):
    if ea.standardmodel_postqcd[i][2] ==True and ea.standardmodel_postqcd[i][0]!='Neutrinos':
        total_bm+=ea.standardmodel_postqcd[i][4][0]
    else:
        pass

for i in range(len(ea.standardmodel_preqcd)):
    if ea.standardmodel_preqcd[i][2] ==True and ea.standardmodel_preqcd[i][0]!='Neutrinos':
        total_bm+=ea.standardmodel_preqcd[i][4][0]
    else:
        pass
    
#Add protons/neutrons
total_bm+=ea.early_proton+ea.early_neutron+ea.early_hydrogen
    
#Entropy Density
for i in range(len(ea.standardmodel_postqcd)):
    if ea.standardmodel_postqcd[i][0]!='Photon' and ea.standardmodel_postqcd[i][0]!='Neutrinos':
        ea.early_s_baryon+=ea.standardmodel_postqcd[i][4][2]
    else:
        pass

for i in range(len(ea.standardmodel_preqcd)):
    if ea.standardmodel_preqcd[i][0]!='Photon' and ea.standardmodel_preqcd[i][0]!='Neutrinos':
        ea.early_s_baryon+=ea.standardmodel_preqcd[i][4][2]
    else:
        pass

total_bm=total_bm+ea.early_proton+ea.early_neutron
total_m=total_bm+ea.early_dm
zero_crossings = np.where(np.diff(np.sign(eps_rel-total_m)))[0]
#zero_crossings = np.where(np.diff(np.sign(standardmodel_preqcd[13][4][0]+standardmodel_postqcd[-1][4][0]-total_m)))[0]

early_zeq=1/early_a[zero_crossings[-1]]-1
theoretical_zeq=(c.om_bm0+c.om_dm0)/(c.om_cmb+c.om_nu)


#Time/Hubble parameter
#SET PROPER TIME OF CMB, DOES NOT CORRELATE TO MODERN UNIVERSE CODE
set_t=time[0] #set_t=380000*yr
early_eta=np.zeros(c.early_length)
early_eta[-1]=eta[decoupling_index]
early_t=np.zeros(c.early_length)
early_t[-1]=set_t#time[decoupling_index]
e_h=h*10**(-3)
e_t=e_h/10
l_temp=10000
eta_temp=np.zeros(l_temp)
time_temp=np.zeros(l_temp)
a_temp=np.zeros(l_temp)
eta_temp[-1]=eta[decoupling_index]
time_temp[-1]=time[decoupling_index]
a_temp[-1]=a[decoupling_index]
dt_array=np.zeros(l_temp)
adot=np.zeros(l_temp)
early_r_ceh=np.zeros(c.early_length)
early_volumeceh=np.zeros(c.early_length)
early_s_ceh=np.zeros(c.early_length)

backwardsindex_array=np.flip(np.arange(0, len(earlytemp)))

backwardsindex_fromdecoupling=np.flip(np.arange(0, c.early_length))

for i in backwardsindex_fromdecoupling:
        epstot=total_m[i]+ea.early_l[i]+eps_rel[i]
        early_hub[i]=np.sqrt(8*np.pi*c.G*epstot/(3*c.c**2))*c.mpc/1000 #Assume H=adot/a
        adot[i]=early_hub[i]*early_a[i]
        if i!=backwardsindex_fromdecoupling[0] and eps_rel[i]>total_m[i]:
            early_t[i]=early_t[i+1]*(early_a[i]/early_a[i+1])**2
            early_eta[i]=early_eta[i+1]*(early_a[i]/early_a[i+1])
        elif i!=backwardsindex_fromdecoupling[0] and eps_rel[i]<total_m[i]:
            early_t[i]=(early_t[i+1]**(2./3.)*early_a[i]/early_a[i+1])**(3./2.)
            early_eta[i]=early_eta[i+1]*(early_a[i]/early_a[i+1])**2
        else:
            pass
        early_r_ceh[i]=early_a[i]*(c.etainf-(early_eta[i]))*c.c
        early_volumeceh[i]=4*np.pi*early_r_ceh[i]**3/3
        early_s_ceh[i]=(early_r_ceh[i]**2*np.pi*c.kb*c.c**3/(c.G*c.hbar))/early_volumeceh[i]
        
#%%

#PBHs

def lifetime(m): #m in kg
    m=m*1000/1e10
    return 407*m**3

def betaprime2f(betaprime):
    gam=0.2
    g_pbh=early.gstar(earlytemp[mev100], ea.standardmodel_preqcd, ea.standardmodel_postqcd)
    h_pbh=1#early_hub[mev100]/100
    etot=total_m[mev100]+ea.early_l[mev100]+eps_rel[mev100]
    edm=ea.early_dm[mev100]
    betam=betaprime/(gam**(1/2)*(g_pbh/106.75)**(-1/4)*(h_pbh/0.67)**(-2))
    f=betam*etot/edm
    return f

def massfunc(option):
    if option==0: #Lower limit with mass frac=1
        sol=10**(-16)
        frac=1
    elif option==1: #Upper limit with mass frac=1
        sol=10**(18.5)
        frac=1
    elif option==2: #Upper limit of PBH size
        sol=10**(21)
        frac=0.03
    elif option==3: #Not possible due to constraints, but will evaporate
        sol=2.5e-19#3.6e-19
        frac=1
    elif option==4: #Largest mass BH to have fully evaporated
        sol=(c.t0/407)**(1/3)*1e10/1000
        frac=1e-7 #Approximate due to evaporation constraints from CMB, EGB, GGB
    return sol*c.msol, frac #Black hole mass in kg, mass fraction of CDM

def dmdt(m): #dm10/dt
    f=1 #Normalized number of emitted particle species for black hole of mass M (ranges from 1 to 15.35)
    return -5.34*10**(-5)*f/m**2 #m in m10

#m=m10*10^10g
#10^15g-10^17g, 10^14g-10^15g

def fnum(m): #m in msol
    if m*c.msol*1000>1e17: #only emits photons
        sol=1
    elif m*c.msol*1000>5e14: #emits electrons and positons as well
        sol=1.569
    else:
        m=c.hbar*c.c**3/(8*np.pi*c.G*c.kb*m) #convert to temperature
        sol=1.569+0.569*(np.exp(-0.0234/m)+6*np.exp(-0.066/m)+3*np.exp(-0.11/m)+3*np.exp(-0.394/m)+3*np.exp(-0.413/m)+3*np.exp(-1.17/m)+3*np.exp(-22/m))+0.963*np.exp(-0.10/m)
    return sol

# def bhmassfun(m): #m in msol
#     m=m*msol*1000 #msol to grams conversion
#     return m**2/fnum(m) #returns in grams

# dm=1e13 #in grams

# m0=5e14/(msol*1000)

# def m_bh(t, m0):
#     rhs_temp=-5.34*1e25*t #units are g^3
#     runningsol=0
#     if runningsol<rhs_temp:
#         runningsol+=

def t_elapsed(option, m_f): #m_f in kg
    m_i=massfunc(option)[0]*1000/1e10 #m_i in m10
    m_f=m_f*1000/1e10 #m_f in m10
    sol=0
    if option==3:
        if m_f>1e4:
            sol+=(m_f**3-(m_i)**3)/(3*2.1)
        else:
            sol+=((1e4)**3-(m_i)**3)/(3*2.1)
            sol+=(m_f**3-(1e4)**3)/(3*15.35)
    elif option==4:
        sol+=(m_f**3-m_i**3)/(3*15.35)
    else:
        if m_f>1e7:
            sol+=(m_f**3-m_i**3)/3
        else:
            sol+=((1e7)**3-m_i**3)/3
            if m_f>1e5:
                sol+=(m_f**3-(1e7)**3)/(3*1.5)
            else:
                sol+=((1e5)**3-(1e7)**3)/(3*1.5)
                if m_f>1e4:
                    sol+=(m_f**3-(1e5)**3)/(3*2.1)
                else:
                    sol+=((1e4)**3-(1e5)**3)/(3*2.1)
                    sol+=(m_f**3-(1e4)**3)/(3*15.35)
    sol=sol/(-5.34*10**(-5))
    return sol #returns time in seconds since PBH creation

def t_hawking(m): #7.1.9 in Wald, Chicago Lectures in Physics
    m=np.array(m)
    return c.hbar*c.c**3/(8*np.pi*c.G*m*c.kb)

# def s_hawking(t, m0): #Absolute entropy as a function of time given original mass in kg and t in seconds
#     t=np.array(t)
#     sol=[]
#     for i in t:
#         if 3*(5.34*1e25/1e9)*i<m0**3:
#             sol.append(16*np.pi*G*kb*m0**2/(3*hbar*c)*(1-(1-3*(5.34*1e25/1e9)*i/m0**3)**(2./3.)))
#         else:
#             sol.append(16*np.pi*G*kb*m0**2/(3*hbar*c))
#     return np.array(sol)


def s_hawking(m, m0): #Absolute entropy as a function of mass given original mass in kg and m in kg
    m=np.array(m)
    return 16*np.pi*c.G*c.kb*(m0**2-m**2)/(3*c.hbar*c.c)


totaltime=np.concatenate((early_t, time))

mass_range0=np.logspace(np.log10(massfunc(0)[0]/(1+1e-8)),np.log10(massfunc(0)[0]))[:-1] #masses in kg for t_elapsed
t_range0=[]
for i in range(len(mass_range0)):
    if t_elapsed(0,mass_range0[i])<lifetime(massfunc(0)[0]):
        t_range0.append(t_elapsed(0,mass_range0[i]))
    else:
        t_range0.append(t_elapsed(0,mass_range0[i]))
        mass_range0[i]=0
t_range0=np.array(t_range0)
mass_range0=np.append(0, mass_range0)
t_range0=np.append(lifetime(massfunc(0)[0]), t_range0)

mass0=[]
time0=[]
index0=[]

for i in np.flip(range(len(t_range0))):
    running_t=0
    running_index=mev100+1
    while running_t<t_range0[i] and running_index<(len(totaltime)-1):
        running_t=totaltime[running_index]-totaltime[mev100]
        running_index+=1
    time0.append(totaltime[running_index])
    index0.append(running_index)
            
mass_range1=np.logspace(np.log10(massfunc(1)[0]/(1+1e-43)),np.log10(massfunc(1))[0])[:-1] #masses in kg for t_elapsed
t_range1=[]
for i in mass_range1:
    t_range1.append(t_elapsed(1,i))
t_range1=np.array(t_range1)
mass1=[]
time1=[]
index1=[]

for i in np.flip(range(len(t_range1))):
    running_t=0
    running_index=mev100+1
    while running_t<t_range1[i] and running_index<(len(totaltime)-1):
        running_t=totaltime[running_index]-totaltime[mev100]
        running_index+=1
    time1.append(totaltime[running_index])
    index1.append(running_index)

mass_range2=np.logspace(np.log10(massfunc(2)[0]/(1+1e-46)),np.log10(massfunc(2))[0])[:-1] #masses in kg for t_elapsed
t_range2=[]
for i in mass_range2:
    t_range2.append(t_elapsed(2,i))
t_range2=np.array(t_range2)
mass2=[]
time2=[]
index2=[]

for i in np.flip(range(len(t_range2))):
    running_t=0
    running_index=mev100+1
    while running_t<t_range2[i] and running_index<(len(totaltime)-1):
        running_t=totaltime[running_index]-totaltime[mev100]
        running_index+=1
    time2.append(totaltime[running_index])
    index2.append(running_index)

#mass_range3=np.logspace(0,np.log10(massfunc(3))[0])
mass_range3=np.logspace(np.log10(massfunc(3)[0]/(1+1e-2)),np.log10(massfunc(3))[0])[:-1] #masses in kg for t_elapsed
t_range3=[]
for i in range(len(mass_range3)):
    if t_elapsed(3,mass_range3[i])<lifetime(massfunc(3)[0]):
        t_range3.append(t_elapsed(3,mass_range3[i]))
    else:
        t_range3.append(t_elapsed(3,mass_range3[i]))
        mass_range3[i]=0
t_range3=np.array(t_range3)
mass_range3=np.append(0, mass_range3)
t_range3=np.append(lifetime(massfunc(3)[0]), t_range3)

mass3=[]
time3=[]
index3=[]

for i in np.flip(range(len(t_range3))):
    running_t=0
    running_index=mev100+1
    while running_t<t_range3[i] and running_index<(len(totaltime)-1):
        running_t=totaltime[running_index]-totaltime[mev100]
        running_index+=1
    time3.append(totaltime[running_index])
    index3.append(running_index)

mass_range4=np.logspace(np.log10(massfunc(4)[0]/(1+1e-1)),np.log10(massfunc(4))[0])[:-1] #masses in kg for t_elapsed
t_range4=[]
for i in range(len(mass_range4)):
    if t_elapsed(4,mass_range4[i])<lifetime(massfunc(4)[0]):
        t_range4.append(t_elapsed(4,mass_range4[i]))
    else:
        t_range4.append(t_elapsed(4,mass_range4[i]))
        mass_range4[i]=0
t_range4=np.array(t_range4)
mass_range4=np.append(0, mass_range4)
t_range4=np.append(lifetime(massfunc(4)[0]), t_range4)

mass4=[]
time4=[]
index4=[]

for i in np.flip(range(len(t_range4))):
    running_t=0
    running_index=mev100+1
    while running_t<t_range4[i] and running_index<(len(totaltime)-1):
        running_t=totaltime[running_index]-totaltime[mev100]
        running_index+=1
    time4.append(totaltime[running_index])
    index4.append(running_index)

def spbh0(eps, option): #eps is CDM energy density, option[0] is BH mass, n is number DENSITY of BHs
    n=eps*massfunc(option)[1]/(massfunc(option)[0]*c.c**2)
    return n*4*np.pi*c.kb*c.G*massfunc(option)[0]**2/(c.c*c.hbar) #returns entropy density of PBHs

def spbh(eps, option, m): #eps is CDM energy density, option[0] is BH mass, n is number DENSITY of BHs
    n=np.array(eps*massfunc(option)[1]/(massfunc(option)[0]*c.c**2))
    return n*4*np.pi*c.kb*c.G*(np.array(m))**2/(c.c*c.hbar), n #returns entropy density of PBHs


totaldm=np.concatenate((ea.early_dm, eps_dm))
totala=np.concatenate((early_a, a))

pbhtype=0

mass_range0=np.flip(mass_range0)
mass0=np.interp(totaltime[mev100:], time0, mass_range0)
mass0=np.concatenate((np.zeros(len(totaltime)-len(mass0)),np.array(mass0)))

early_pbh0=np.zeros(c.early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh0[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass0[mev100:])[0]
early_pbh0_noevap=np.zeros(c.early_length+l)
early_pbh0_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass0[-1])[0]
hawking0=np.zeros(c.early_length+l)
hawking0[(mev100):]=s_hawking(mass0[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass0[mev100:])[1]

figpbh, axpbh = plt.subplots(1,1)
axpbh.scatter(np.log10(totala), np.log10(early_pbh0), c='r', marker='x')
axpbh.scatter(np.log10(totala), np.log10(early_pbh0_noevap), c='b', marker='+')

# for i in np.arange(mev100+1, early_length):
#     early_pbh0[i]=early_pbh0[i-1]+(dmdt(massfunc(pbhtype)[0]*10**(-7))*10**(7)* #m_i=m_(i-1)+dm

#early_pbh0[early_length]=early_pbh0[early_length-1]+(dmdt(massfunc(pbhtype)[0])/early_hub[early_length-1])*early_a[early_length-1]*(a[decoupling_index]-early_a[early_length-1])

# for i in np.arange(early_length, l+early_length):
#     late_index=decoupling_index+i-early_length-1
#     early_pbh0[i]=early_pbh0[i-1]+(dmdt(massfunc(pbhtype)[0])/hub[late_index])*a[late_index]*(a[late_index+1]-a[late_index])

early_wimp0=ea.early_dm*(1-massfunc(pbhtype)[1])



pbhtype=1

mass_range1=np.flip(mass_range1)
mass1=np.interp(totaltime[mev100:], time1, mass_range1)
mass1=np.concatenate((np.zeros(len(totaltime)-len(mass1)),np.array(mass1)))

early_pbh1=np.zeros(c.early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh1[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass1[mev100:])[0]
early_pbh1_noevap=np.zeros(c.early_length+l)
early_pbh1_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass1[-1])[0]
#early_wimp1=early_dm*(1-massfunc(pbhtype)[1])
hawking1=np.zeros(c.early_length+l)
hawking1[(mev100):]=s_hawking(mass1[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass1[mev100:])[1]

pbhtype=2

mass_range2=np.flip(mass_range2)
mass2=np.interp(totaltime[mev100:], time2, mass_range2)
mass2=np.concatenate((np.zeros(len(totaltime)-len(mass2)),np.array(mass2)))

early_pbh2=np.zeros(c.early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh2[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass2[mev100:])[0]
early_pbh2_noevap=np.zeros(c.early_length+l)
early_pbh2_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass2[-1])[0]
#early_wimp2=early_dm*(1-massfunc(pbhtype)[1])
hawking2=np.zeros(c.early_length+l)
hawking2[(mev100):]=s_hawking(mass2[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass2[mev100:])[1]

pbhtype=3

mass_range3=np.flip(mass_range3)
mass3=np.interp(totaltime[mev100:], time3, mass_range3)
mass3=np.concatenate((np.zeros(len(totaltime)-len(mass3)),np.array(mass3)))

early_pbh3=np.zeros(c.early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh3[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass3[mev100:])[0]
early_pbh3_noevap=np.zeros(c.early_length+l)
early_pbh3_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, massfunc(3)[0])[0]
hawking3=np.zeros(c.early_length+l)
hawking3[(mev100):]=s_hawking(mass3[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass3[mev100:])[1]

figpbh3, axpbh3 = plt.subplots(1,1)
axpbh3.scatter(np.log10(totala), np.log10(early_pbh3), c='c', marker='x')
axpbh3.scatter(np.log10(totala), np.log10(early_pbh3_noevap), c='y', marker='+')

pbhtype=4

mass_range4=np.flip(mass_range4)
mass4=np.interp(totaltime[mev100:], time4, mass_range4)

mass4=np.concatenate((np.zeros(len(totaltime)-len(mass4)),np.array(mass4)))

early_pbh4=np.zeros(c.early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh4[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass4[mev100:])[0]
early_pbh4_noevap=np.zeros(c.early_length+l)
early_pbh4_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, massfunc(4)[0])[0]
hawking4=np.zeros(c.early_length+l)
hawking4[(mev100):]=s_hawking(mass4[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass4[mev100:])[1]

figpbh4, axpbh4 = plt.subplots(1,1)
axpbh4.scatter(np.log10(totala), np.log10(early_pbh4), c='m', marker='x')
axpbh4.scatter(np.log10(totala), np.log10(early_pbh4_noevap), c='k', marker='+')


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

#early_r_ceh=np.zeros(early_length)
#early_s_ceh=np.zeros(early_length)
#early_r_ceh[l-i]=a[l-i]*(etainf-(eta0-h*(i-1)))*c
#early_s_ceh[l-i]=(r_ceh[l-i]**2*np.pi*kb*c**3/(G*hbar))/volumeceh[l-i]

#%%

#Entropy Densities in Early Universe

print("Energy at Decoupling:  Early Code, Late Code")
print("-------------------")
print("Cosmic Microwave Background: %10.2E, \t(%10.2E)" % (ea.standardmodel_postqcd[-1][4][0][-1], eps_cmb[0]))
print("CMB Temp                   : %10.2E, \t(%10.2E)" % (ea.standardmodel_postqcd[-1][4][1][-1]*c.ev2K, t_cmb[0]))
print("Primordial Neutrinos       : %10.2E \t(%10.2E)" % (ea.standardmodel_postqcd[-3][4][0][-1], eps_nu[0]))
print("Neutrino Temp              : %10.2E \t(%10.2E)" % (ea.standardmodel_postqcd[-3][4][1][-1]*c.ev2K, t_nu[0]))
print("Dark Matter                : %10.2E \t(%10.2E)" % (total_m[-1], eps_dm[0]))



#%%

#Significant features for plots
strpeak_arg=np.argmax(abs(bh.strdot(hub)))
lambdadom_arg=np.argmin(abs(eps-2*eps_l))
zeq_arg=np.argmin(abs(a-1/(1+c.zeq)))
zdec_arg=np.argmin(abs(a-1/(1+c.zdec)))
zrec_arg=np.argmin(abs(a-1/(1+c.zrec)))

#use chemical potential formulae for hydrogen and helium only when going past redshift 20
#write chemical potential as a function of number density nd temperature?
#baryon number desnity scales as a^-3, can use photon baryon ratio which is invariant and CMB number dens
#1/(length^3*temp**3/2*mass**3/2) means (G*hbar/c^3)^3/2*((hbar*c/G)^(1/2)*c^2/kb)^(3/2)*(hbar*c/G)^(3/4)
#G^(3/2-1/2*3/2-3/4)*hbar^(3/2+1/2*3/2+3/4)*c^(-3*3/2+(1/2+2)*3/2+3/4)*kb^(-3/2)
#G^0*hbar^(3)*c^0*kb^(-3/2)
'''
hbar=m^2*kg/s
kb=m^2*kg/(s^2*K)
hbar^3/kb^(3/2)=(m^6*kg^3/s^3)/(m^3*kg^3/2/(s^3*K^3/2))
=m^3*kg^3/2*K^3/2
'''
        
"""
Testing
"""

s_smbhmf=np.flip(s_smbhmf)
smmarray=np.logspace(8, 17, num=100)
peaksdensity=[]
for i in s_smbhmf:
    massarg=np.argmax(i)
    peaksdensity.append(np.log10(smmarray[massarg]))

smbhfg, smbhax = plt.subplots(1,1)
smbhax.scatter(np.log(a[1:]), peaksdensity)

#more smbh
ldapeak=2.5
betal=0.3
mdotcrit=0.01
etarad0=0.11
Leddsol=1.3*10**(38) #erg/s
sexp=0.5
delta=0.5 #duty cycle parameter

def unnormzigzag(lda):
    return (lda/ldapeak)**(-betal)*np.exp(-lda/ldapeak)

cl=etarad0*np.log(10)*mdotcrit*1.3*10**(38)/(integrate.quad(unnormzigzag, 10**(-4), ldapeak)[0]*0.1*delta*Leddsol*(1-etarad0))

def zigzag(lda):
    return unnormzigzag(lda)*cl

def zigzagint():
    intsol=integrate.quad(zigzag, 10**(-4), ldapeak)[0]
    return intsol/np.log(10)

def etarad(mdt):
    if m>=mdotcrit:
        sol=etarad0
    else:
        sol=etarad0*(m/mdotcrit)**sexp
    return sol

def integrand(lda, m, mdt):
    return zigzag(lda)*(m/c.msol)*lda*Leddsol*(1-etarad(mdt))/(etarad(mdt)*c.c**2)

def mdot(m):
    Medd=1.3*10**(38)*m/(0.1*c.msol*c.c**2)
    # mdt0=1
    rhsmdot0=zigzagint*delta*(m/c.msol)*Leddsol*(1-etarad0)/(etarad0*c.c**2)
    rhsmratio0=rhsmdot0/Medd
    # mdottemp=sol/Medd
    # if sol<mdotcrit:
    #     mdt0=0
    #     sol=integrate.quad(integrand, 10**(-4), ldapeak, arg=(m, mdt0))
    # else:
    #     pass
    return rhsmratio0#sol



#calculate the mass input that results in mdotcrit, then add dM to that mass and use mdotcrit for mdot, calculate the new mdot, then use that new mdot for the next mass, M+dM

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


print("Present Day Entropy:  S[k] from this code (Egan & Lineweaver 2009)")
print("-------------------")
print("Cosmic Event Horizon       : %10.2E \t(2.6E+122)" % (s_ceh[-1]*volumeceh[-1]/c.kb))
print("Supermassive Black Holes   : %10.2E \t(1.2E+103)" % (s_smbh[-1]*volumeceh[-1]/c.kb))
print("Stellar Mass Black Holes   : %10.2E \t(2.2E+96)" % (s_bh[-1]*volumeceh[-1]/c.kb))
print("Cosmic Microwave Background: %10.2E \t(2.03E+88)" % (s_cmb[-1]*volumeceh[-1]/c.kb))
print("Primordial Neutrinos       : %10.2E \t(1.93E+88)" % (s_nu[-1]*volumeceh[-1]/c.kb))
print("Gas and Dust               : %10.2E \t(2.7E+80)" % (s_bm[-1]*volumeceh[-1]/c.kb))