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

plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=12)



"""
Constants in MKS
"""

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

#Numerical integration parameters
h=1e14#step size in eta, da=sqrt(8*pi*G*eps/3)*a^2*d(eta) from 1.70 Mukhanov
l=100000-86181#4830-224 #array size

#%%
"""
Initialize arrays for scale factor, energy density, entropy density, temperature
"""

time=np.zeros(l) #Cosmic time
time[l-1]=t0

eta=np.zeros(l) #Conformal time
eta[l-1]=eta0

a=np.zeros(l) #Scale factor
a[l-1]=1
a_ceharg=2982

hub=np.zeros(l) #Hubble constant
hub[l-1]=h0

r_p=np.zeros(l) #Particle Horizon (Observable Universe)
r_p[l-1]=eta0*c

volume=np.zeros(l) #Volume uses particle horizon
volume[l-1]=4*np.pi*(r_p[l-1])**3/3

r_ceh=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh[l-1]=c*(etainf-eta0)
r_ceh_h=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh_h[l-1]=c*(etainf_h-eta0_h)
r_ceh_l=np.zeros(l) #Radius of Cosmic Event Horizon
r_ceh_l[l-1]=c*(etainf_l-eta0_l)

volumeceh=np.zeros(l) #Volume calculated with CEH
volumeceh[l-1]=4*np.pi*(r_ceh[l-1])**3/3
volumeceh_h=np.zeros(l)
volumeceh_h[l-1]=4*np.pi*(r_ceh_h[l-1])**3/3
volumeceh_l=np.zeros(l)
volumeceh_l[l-1]=4*np.pi*(r_ceh_l[l-1])**3/3

s_ceh=np.zeros(l)
stot_ceh=np.zeros(l)
s_ceh[l-1]=(r_ceh[l-1]**2*np.pi*kb*c**3/(G*hbar))/volumeceh[l-1]
stot_ceh[l-1]=(r_ceh[l-1]**2*np.pi*kb*c**3/(G*hbar))
s_ceh_h=np.zeros(l)
stot_ceh_h=np.zeros(l)
s_ceh_h[l-1]=(r_ceh_h[l-1]**2*np.pi*kb*c**3/(G*hbar))/volumeceh_h[l-1]
stot_ceh_h[l-1]=(r_ceh_h[l-1]**2*np.pi*kb*c**3/(G*hbar))
s_ceh_l=np.zeros(l)
stot_ceh_l=np.zeros(l)
s_ceh_l[l-1]=(r_ceh_l[l-1]**2*np.pi*kb*c**3/(G*hbar))/volumeceh_l[l-1]
stot_ceh_l[l-1]=(r_ceh_l[l-1]**2*np.pi*kb*c**3/(G*hbar))

eps_bm=np.zeros(l) #Baryons
eps_bm0=eps_c0*om_bm0
eps_bm[l-1]=eps_bm0
t_bm=np.zeros(l)
t_bm[l-1]=0.025468041207723818 #This came from old ASU code from Judd Bowman?
s_bm=np.zeros(l)
#Eagle data
bm_eagleresults=[148.4053081304242, 84.3039732671589, 418.17238398806165, 853.2427715295006, 1572.864365719125, 1810.0005155900126, 1759.1301851009557, 2699.516883529645, 2110.990845022963, 2226.961377352639, 3126.284474107758, 4430.343761110442, 6127.405729410453, 4862.621531291029, 7113.963501171243, 8153.852287086782, 6080.247274201631, 8615.433172615767, 1927.9765390416364, 8000.553030185015, 1305.644296343521, 2968.6629285091212, 824.4928913315442, 1269.6081292868707, 464.85246944704045, 280.57777072388836, 233.8795437672676, 232.7706791309389, 235.5692882942148]
bmz=np.array([0.00, 0.10, 0.18, 0.27, 0.37, 0.50, 0.62, 0.74, 0.87, 1.00, 1.26, 1.49, 1.74, 2.01, 2.24, 2.48, 3.02, 3.53, 3.98, 4.49, 5.04, 5.49, 5.97, 7.05, 8.07, 8.99, 9.99, 15.13, 20.00])
bma=1/(bmz+1)
s_bm[l-1]=bm_eagleresults[0]*kb

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
s_bhhigh=np.zeros(l)
s_bhlow=np.zeros(l)

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

eps_cmb=np.zeros(l) #Cosmic Microwave Background photons
eps_cmb0=eps_c0*om_cmb
eps_cmb[l-1]=eps_cmb0
t_cmb=np.zeros(l)
t_cmb0=2.7255 #K, Fixen 2009
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

eps_smbh=np.zeros(l) #Supermassive black holes
s_smbh=np.zeros(l)
s_smbhhigh=np.zeros(l)
s_smbhlow=np.zeros(l)

eps_c=np.zeros(l) #Critical energy density
eps_c[l-1]=eps_c0

eps=np.zeros(l) #Total energy
eps[l-1]=eps_dm0+eps_bm0+eps_cmb0+eps_nu0+eps_l0
s=np.zeros(l)
#%%
"""
Functions for a' and a'', conformal time derivatives
"""

def ap(a, epst):
    return np.sqrt(8*np.pi*G*epst*a**4/(3.0*c**2)-k*c**2*a**2)

def ap2(a, epsbm, epsdm, epsl):
    return 4*np.pi*G*(epsdm+epsbm+4*epsl)*a**3/(3*c**2)-k*a

"""
Stellar mass black hole number density calculations
Uses Chabrier IMF, assume black hole progenitors are greater than 25 Msol
Progenitor to remnant mass function crudely approximated from Fryer & Kalogera, 2001
https://arxiv.org/pdf/astro-ph/9911312.pdf
"""

#Peak SFR is around 5-6 z
def strdot(h): #Time derivative of stellar mass density, h is Hubble constant in (km/s)/Mpc
    x=(h/h0)**(2./3.)
    sol=strdot0*x**2/(1+0.012*(x-1)**3*np.exp(0.041*x**(7./4.))) #Hernquist & Springel
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
    else:
        alpha=-2.35
    return m**(alpha+1) #multiply by dlogM to return value proportional to dn

def imfhigh(m): #Initial mass function, redshift independent, m in msol
    if m<0.5:
        alpha=-1.35
    else:
        alpha=-2.35+0.65
    return m**(alpha+1) #multiply by dlogM to return value proportional to dn in 

def imflow(m): #initial mass function redshift independent, m in msol
    if m<0.5:
        alpha=-1.35
    else:
        alpha=-2.35-0.35
    return m**(alpha+1) #multiply by dlogM to get proportional to dn

def norm(epsstr): #normalization constant between rho_str(z)*1 million yrs (1 star formation event) and imf/pmf integral
    sol=1.70939 #Integrated in Mathematica from 0 msol to 300 msol
    a=epsstr/(sol*msol*c**2)
    return a

def normhigh(epsstr):
    solhigh=7.13364 #from +0.65
    ahigh=epsstr/(solhigh*msol*c**2) 
    return ahigh

def normlow(epsstr):
    sollow=1.29284 #from -0.35
    alow=epsstr/(sollow*msol*c**2)
    return alow

#Stellar mass (in Msol) to lifetime

def tstar(m):
    #sol=10**(10)*m**(-2.5)
    sol=10**(10.015-3.461*np.log10(m)+0.8157*(np.log10(m)**2)) #elmegreen 2007 paper, eq2
    return sol

#Coefficient table [logN, logM, alpha, logNG, logMG, sigmaG]
z0=[6.078, 0.704, -2.717, 3.496, 1.808, 0.1846]
z1=[5.887, 0.709, -2.785, 3.304, 1.843, 0.173]
z2=[5.592, 0.713, -2.823, 3.008, 1.866, 0.165]
z4=[4.796, 0.747, -2.782, 2.101, 1.952, 0.132]
z6=[4.112, 0.785, -2.718, 1.359, 2.012, 0.107]
z8=[3.457, 0.816, -2.660, 0.685, 2.046, 0.091]
z10=[2.897, 0.831, -2.623, 0.113, 2.059, 0.0841]
coefficients=np.array([z10, z8, z6, z4, z2, z1, z0])

def bhmf(m, zmf): #N's in Mpc^-3, M's in Msol
    logN, logM, alphamf, logNG, logMG, sigmaG = coefficients[zmf]
    logN=10**(logN)
    logM=10**(logM)
    logNG=10**(logNG)
    logMG=10**(logMG)
    return (logN*(m/logM)**(1-alphamf)*np.exp(-m/logM)+logNG*(2*np.pi*sigmaG**2)**(-0.5)*np.exp(-(np.log(m)-np.log(logMG))**2/(2*sigmaG**2)))/(mpc**3)

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
#%%
#Present day SMBH entropy
runningtot=0
runningtothigh=0
runningtotlow=0
iterations=100
jstart=10**(8.-((17-8)/iterations))
for j in np.logspace(8, 17, num=iterations): #j (DM mass) is in solar masses
    tempu=xi2*(j-jstart)
    dn=rhs(j, 0, eps_bm[-1], eps_dm[-1], eps_c[-1], tempu)*(-h)*ap(1,eps[-1]) #should there be a negative in front of the h?  maybe...?
    if dn<0:
        dn=0
    dnhigh=dn*1.2
    dnlow=dn*0.8
    smbh_mass=10**(1.55*np.log10(j/(1e13))+8.01)*msol #smbh mass in kg
    smbh_masshigh=10**(1.6*np.log10(j/(1e13))+8.05)*msol
    smbh_masslow=10**(1.5*np.log10(j/(1e13))+7.97)*msol
    smbh_dshigh=4*np.pi*kb*G*smbh_masshigh**2/(c*hbar)
    smbh_dslow=4*np.pi*kb*G*smbh_masslow**2/(c*hbar)
    smbh_ds=4*np.pi*kb*G*smbh_mass**2/(c*hbar)
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
specmass=[0.5109989461, 938.272081, 3.01603*amu, 2.014*amu, 4.00260*amu] #MeV (PDG, NIST)
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
    y=a[l-i+1]
    a[l-i]=y-h*ap(y,eps[l-i+1])#-0.5*h**2*ap2(y,eps_bm[l-i+1],eps_dm[l-i+1],eps_l[l-i+1])/2
    
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
    r_ceh[l-i]=a[l-i]*(etainf-(eta0-h*(i-1)))*c
    r_ceh_h[l-i]=a[l-i]*(etainf_h-(eta0_h-h*(i-1)))*c
    r_ceh_l[l-i]=a[l-i]*(etainf_l-(eta0_l-h*(i-1)))*c
    volumeceh[l-i]=4*np.pi*r_ceh[l-i]**3/3
    volumeceh_h[l-i]=4*np.pi*r_ceh_h[l-i]**3/3
    volumeceh_l[l-i]=4*np.pi*r_ceh_l[l-i]**3/3
    volume[l-i]=4*np.pi*r_p[l-i]**3/3
    eps_bm[l-i]=eps_bm[l-i+1]*(a[l-i+1]/a[l-i])**3
    eps_dm[l-i]=eps_dm[l-i+1]*(a[l-i+1]/a[l-i])**3
#    eps_mat[l-i]=eps_dm[l-i]+eps_bm[l-i]
    eps_cmb[l-i]=eps_cmb[l-i+1]*(a[l-i+1]/a[l-i])**4
    eps_nu[l-i]=eps_nu[l-i+1]*(a[l-i+1]/a[l-i])**4
    eps_l[l-i]=eps_l0
#    eps_rad[l-i]=eps_nu[l-i]+eps_cmb[l-i]
    eps[l-i]=eps_dm[l-i]+eps_cmb[l-i]+eps_nu[l-i]+eps_l[l-i]+eps_bm[l-i]
    hub[l-i]=np.sqrt(8*np.pi*G*eps[l-i]/(3*c**2))*mpc/1000
    if (eps_bm[l-i+1]+eps_dm[l-i+1])>eps_l[l-i+1]:
        time[l-i]=time[l-i+1]*(a[l-i]/a[l-i+1])**(3./2.)
    else:
        time[l-i]=(np.log(a[l-i]/a[l-i+1])+hub[l-i+1]*time[l-i+1])/hub[l-i] #Assume H=a'/a^2
#    eps_str[l-i]=eps_str[l-i+1]-strdot(hub[l-i+1])*h*a[l-i+1]*c**2
    #comp=32*sigmat*sigmasb*a[l-i+1]*(t_cmb[l-i+1]**4)*xe*(t_bm[l-i+1]-t_cmb[l-i+1])/(3*hub[l-i+1]*me*(c**2)*(1+xe+xhe)*np.sqrt(8*np.pi*G*eps[l-i+1]/3))
    t_cmb[l-i]=((m_pl*c**2*l_pl)**3*eps_cmb[l-i]*15/(np.pi**2))**(1./4.)/kb #mukhanov exact for massless relativistic bosons (chemical potential=mass=0)
    if t_cmb[l-i]/ev2K>0.27 and switch==False:
        decoupling_index=l-i
        switch=True
    else:
        pass
    t_nu[l-i]=t_cmb[l-i]*(4./11.)**(1./3.) #After electron positron annihilation
    #t_bm[l-i]=t_bm[l-i+1]+h*2*(hub[l-i+1]*1000/mpc)*t_bm[l-i+1]*a[l-i+1] #Adiabatic cooling
    eps_c[l-i]=c**2*3*(hub[l-i]*1000/mpc)**2/(8*np.pi*G)
    z=1/a[l-i]-1
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
        tempu=xi2*(j-jstart)
        dn=rhs(j, z, eps_bm[l-i], eps_dm[l-i], eps_c[l-i], tempu)*(-h)*ap(a[l-i],eps[l-i]) #should there be a negative in front of the h?  maybe...?
        if dn<0:
            dn=0
        dnhigh=dn*1.2
        dnlow=dn*0.8
        smbh_mass=10**(1.55*np.log10(j/(1e13))+8.01)*msol #smbh mass in kg
        smbh_masshigh=10**(1.6*np.log10(j/(1e13))+8.05)*msol
        smbh_masslow=10**(1.5*np.log10(j/(1e13))+7.97)*msol
        smbh_dshigh=4*np.pi*kb*G*smbh_masshigh**2/(c*hbar)
        smbh_dslow=4*np.pi*kb*G*smbh_masslow**2/(c*hbar)
        smbh_ds=4*np.pi*kb*G*smbh_mass**2/(c*hbar)
        runningtot=runningtot+smbh_ds*dn #dn is in #/volume already I think
        runningtothigh=runningtothigh+smbh_dshigh*dnhigh
        runningtotlow=runningtotlow+smbh_dslow*dnlow
        smftemp.append(smbh_ds*dn)
        jstart=j
    s_smbhmf.append(smftemp)
        
    s_smbh[l-i]=runningtot
    s_smbhhigh[l-i]=runningtothigh
    s_smbhlow[l-i]=runningtotlow
    s_cmb[l-i]=2*np.pi**2*kb**4*2*t_cmb[l-i]**3/(45*c**3*hbar**3)
    s_nu[l-i]=2*np.pi**2*kb**4*6*7*t_nu[l-i]**3/(45*c**3*hbar**3*8) #Lineweaver 2009
#    s_rad[l-i]=s_nu[l-i]+s_cmb[l-i]
    stot_ceh[l-i]=(r_ceh[l-i]**2*np.pi*kb*c**3/(G*hbar))
    s_ceh[l-i]=(r_ceh[l-i]**2*np.pi*kb*c**3/(G*hbar))/volumeceh[l-i]
    stot_ceh_h[l-i]=(r_ceh_h[l-i]**2*np.pi*kb*c**3/(G*hbar))
    s_ceh_h[l-i]=(r_ceh_h[l-i]**2*np.pi*kb*c**3/(G*hbar))/volumeceh_h[l-i]
    stot_ceh_l[l-i]=(r_ceh_l[l-i]**2*np.pi*kb*c**3/(G*hbar))
    s_ceh_l[l-i]=(r_ceh_l[l-i]**2*np.pi*kb*c**3/(G*hbar))/volumeceh_l[l-i]
#%%
print("\n Now calculating stellar energy density, stellar mass black holes, and baryons... \n")
#Stellar energy density plotting
z20=np.argmin(abs(1./a-21))
z30=np.argmin(abs(1./a-31))
for zed in np.arange(z30,l):
    eps_str[zed]=eps_str[zed-1]+strdot(hub[zed-1])*h*a[zed-1]*c**2

#Stellar mass black hole number density tracking, typically at 8.05E+97
mstart=10**(1.3-((2-1.3)/1000))
mstart=10**(1.398-((2-1.3)/1000)) #start at M=25 Msol, s=8.05e+97 mstart has negligible change, most comes from high mass BHs
for i in np.arange(z30,l-1):
    for m in np.logspace(1.398, 2, num=1000):
        mstep=(m-mstart)/(m*np.log(10)) #dlogM
        den=norm(eps_str[i])*imf(m)*mstep #dn - number density of progenitors between mass M and M+dM
        denhigh=normhigh(eps_str[i])*imfhigh(m)*mstep
        denlow=normlow(eps_str[i])*imflow(m)*mstep
        #a*deta=dt
        #t=int from a[i] to a[i+steps] of a*deta
        intg=a[i]*h
        steps=1
        while(tstar(m)>intg):
            intg+=a[i+steps]*h
            steps+=1
        if (i+steps)<l:
            s_bh[i+steps]+=den*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
            s_bhhigh[i+steps]+=denhigh*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
            s_bhlow[i+steps]+=denlow*(4*np.pi*kb*G/(c*hbar))*(prog2rem(m)*msol)**2
        mstart=m

z=[10, 8, 6, 4, 2, 1, 0]
zarg=np.arange(len(z))
s_bhinterp=np.zeros(len(zarg))
s_bhinterp0=[]
constcheck=[]
bhmfcheck=[]
mstepcheck=[]
bhindex=[]
mf=[]

for zed in zarg:
    mftemp=[]
    mstart=10**(0.69897-((2.20412-0.69897)/1000)) #start atm=5 to m=160
    bhindex.append(np.argmin(abs(a-(1/(1+np.array(z[zed]))))))
    for m in np.logspace(0.69897, 2.20412, num=1000):
        mstep=(m-mstart)/(m*np.log(10)) #dlogM
        bhmfcheck.append(bhmf(m, zed))
        constcheck.append((4*np.pi*kb*G/(c*hbar))*(m*msol)**2)
        mstepcheck.append(mstep)
        mftemp.append(bhmf(m, zed)*mstep*(4*np.pi*kb*G/(c*hbar))*(m*msol)**2)
        s_bhinterp[zed]+=bhmf(m, zed)*mstep*(4*np.pi*kb*G/(c*hbar))*(m*msol)**2
        mstart=m
    mf.append(mftemp)

abh=a[bhindex]
#s_bh[bhindex[0]:]=np.interp(a[bhindex[0]:], abh, s_bhinterp)

marray=np.logspace(0.69897, 2.20412, num=1000)
peaksbdensity=[]
for i in mf:
    massarg=np.argmax(i)
    peaksbdensity.append(np.log10(marray[massarg]))
bhfg, bhax = plt.subplots(1,1)
bhax.scatter(z, peaksbdensity)

#%%
#Baryons
z20=np.argmin(abs(1./a-21))
s_bm[z20:]=np.interp(a[z20:], np.flip(bma), np.flip(bm_eagleresults))*kb
#s_bm[l-i]=bm_eagleresults[-1]*kb
deg=2
n_b=photonbaryoneta*cmbnumdens/a**3 #Baryon number density in m^-3

# https://arxiv.org/pdf/astro-ph/9909275.pdf
# https://arxiv.org/pdf/astro-ph/9912182.pdf
ar=4*sigmasb/c
def dtm(tr, tm, ne, nb, hubc, a):
    lambdaterms=0
    sol=8*sigmat*ar*tr**4*ne*(tm-tr)/(3*hubc*me*c*nb)+2*tm+2*lambdaterms/(3*kb*nb*h)
    return sol*a #dT/dz

def Xfrac(t): #https://people.ast.cam.ac.uk/~pettini/Intro%20Cosmology/Lecture09.pdf
    q=ionization_energy[0]
    s=3.84*photonbaryoneta*(kb*t/(me*c**2))**1.5*np.exp(q/(kb*t))
    return 1/(1+s)

adecarg=np.argmin(abs(a-1/(1+zdec)))
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

s_bm=n_b*(5./2.-np.log((hbar**3/kb**(3./2.))*n_b*(2*np.pi/(t_bm*bmass))**(3./2.)/(deg*(1+15.*t_bm*kb/(8.*(bmass*c**2))))))*kb

s_bm[z20:]=np.interp(a[z20:], np.flip(bma), np.flip(bm_eagleresults))*kb
#%%

'''
Early Universe
We approximate interaction rates from dimensional analysis of the cross section and temperature dependence of the number density, ignoring v (Gamma=sigma*n*v)
'''
print("\n Now calculating pre-CMB era... \n")

early_length=1000
earlytemp=np.logspace(np.log10(t_cmb[decoupling_index]/ev2K),11, early_length) #eV

#Format of these arrays is [[eps1, temp1, s1], [eps2, tem2, s2], ...]
early_pretop=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prebottom=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_precharm=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prestrange=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_preup=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_predown=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_pregluon=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_pretau=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_premuon=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_preelectron=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_preneutrino=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prew=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prez=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prephoton=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_prehiggs=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]

early_posttop=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postbottom=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postcharm=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_poststrange=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postup=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postdown=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postgluon=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_posttau=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postmuon=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postelectron=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postneutrino=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postw=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postz=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postphoton=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_posthiggs=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postchargedpion=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]
early_postneutralpion=[np.zeros(early_length), np.zeros(early_length), np.zeros(early_length)]

early_baryon=np.zeros(early_length)
early_dm=np.zeros(early_length)
early_l=np.zeros(early_length)
early_proton=np.zeros(early_length)
early_neutron=np.zeros(early_length)
early_hydrogen=np.zeros(early_length)
protontemp=np.zeros(early_length)
neutrontemp=np.zeros(early_length)
early_mup=np.zeros(early_length)
early_mun=np.zeros(early_length)

early_s_proton=np.zeros(early_length)
early_s_neutron=np.zeros(early_length)
early_s_hydrogen=np.zeros(early_length)
early_s_baryon=np.zeros(early_length)
early_s_dm=np.zeros(early_length)
early_dm[-1]=eps_dm[decoupling_index]
early_l[-1]=eps_l[decoupling_index]
early_s_l=np.zeros(early_length)


#(Particle name, mass in eV, fermion? (True if yes), DoF, data arrays)
standardmodel_preqcd=[['Top', 173*1e9, True, 12, early_pretop],
                        ['Bottom', 4*1e9, True, 12, early_prebottom],
                        ['Charm', 1*1e9, True, 12, early_precharm],
                        ['Strange', 100*1e6, True, 12, early_prestrange],
                        ['Down', 5*1e6, True, 12, early_predown],
                        ['Up', 2*1e6, True, 12, early_preup],
                        ['Gluon', 0, False, 16, early_pregluon],
                        ['Tau', 1777*1e6, True, 4, early_pretau],
                        ['Muon', 106*1e6, True, 4, early_premuon],
                        ['Electron', 511*1e3, True, 4, early_preelectron],
                        ['Neutrinos', 0.6, True, 6, early_preneutrino],
                        ['Ws',80*1e9, False, 6, early_prew],
                        ['Z', 91*1e9, False, 3, early_prez],
                        ['Photon', 0, False, 2, early_prephoton],
                        ['Higgs', 125*1e9, False, 1, early_prehiggs]
                        ]
standardmodel_postqcd=[['Top', 173*1e9, True, 12, early_posttop],
                        ['Higgs', 125*1e9, False, 1, early_posthiggs],
                        ['Z', 91*1e9, False, 3, early_postz],
                        ['Ws',80*1e9, False, 6, early_postw],
                        ['Bottom', 4*1e9, True, 12, early_postbottom],
                        ['Tau', 1777*1e6, True, 4, early_posttau],
                        ['Charm', 1*1e9, True, 12, early_postcharm],
                        ['Pi+-', 139*1e6, False, 2, early_postchargedpion],
                        ['Pi0', 135*1e6, False, 1, early_postneutralpion],
                        ['Muon', 106*1e6, True, 4, early_postmuon], #NEUTRINO WEIGHT????
                        ['Neutrinos', 1*1e6, True, 6, early_postneutrino],
                        ['Electron', 511*1e3, True, 4, early_postelectron],
                        ['Photon', 0, False, 2, early_postphoton]
                        ]

ewphase=100*1e9 #Electroweak phase transition in eV, W and Z bosons become massive and weak interaction cross section changes
qcdphase=150*1e6 #QCD phase transition in eV, quark/gluon interaction becomes important and quarks are bound by gluons in baryons and mesons
nu_dec=0.8*1e6 #Neutrino decoupling energy from Baumann notes (0.8 MeV)
emphase=1 #Electromagnetic phase transition

e_dec=standardmodel_postqcd[11][1]

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

def gstar(t):
    sol=0
    if t>qcdphase: #When t is greater than QCD phase transition, quarks are unbound and pions don't exist, and all species are coupled
        for i in standardmodel_preqcd:
            if i[1]<(t/6.): #if true, then mass is less than temp
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            else: #case when non-relativistic
                pass
    else: #Below the QCD phase transition, use the post-array with bound quarks in pions
        for i in standardmodel_postqcd:
            if i[1]<(t/6.) and i[0]!='Neutrinos': #relativistic
                # if i[0]=='Neutrinos' and t<nu_dec:
                #     sol+=7./8.*i[3]*(t_i/t)**4
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t>nu_dec:
                sol+=i[3]*7./8. #Eventually, you can replace with with temperature
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t<nu_dec:
                sol+=neff*2*7./8.*(4./11.)**(4./3.) #Eventually, you can replace with with temperature
            else: #case when non-relativistic or decoupled neutrinos
                pass
    return sol
            
def gstarS(t):
    sol=0
    if t>qcdphase: #When t is greater than QCD phase transition, quarks are unbound and pions don't exist, and all species are coupled
        for i in standardmodel_preqcd:
            if i[1]<(t/6.): #if true, then mass is less than temp
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            else: #case when non-relativistic
                pass
    else: #Below the QCD phase transition, use the post-array with bound quarks in pions
        for i in standardmodel_postqcd:
            if i[1]<(t/6.) and i[0]!='Neutrinos': #relativistic
                # if i[0]=='Neutrinos' and t<nu_dec:
                #     sol+=7./8.*i[3]*(t_i/t)**4
                if i[2]==True: #if True, then it is a fermion
                    sol+=i[3]*7./8.
                else: #boson
                    sol+=i[3]
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t>nu_dec:
                sol+=i[3]*7./8. #Eventually, you can replace with with temperature
            elif i[1]<(t/6.) and i[0]=='Neutrinos' and t<nu_dec:
                sol+=neff*2*7./8.*(4./11.) #Eventually, you can replace with with temperature
            else: #case when non-relativistic or decoupled neutrinos
                pass
    return sol

def nonrel_energydensity(g, m, T): #in eV for mass and temp
    T=T*ev2K
    m=m*ev #ev is one electronvolt in kg
    return g*(m*c**2/l_pl**3)*(m*c**2*kb*T/(2*np.pi*(m_pl*c**2)**2))**(3./2.)*np.exp(-m*c**2/(kb*T))*(1+15*kb*T/(8*m*c**2))#g*m*(m*T/(2*np.pi))**(3./2.)*np.exp(-m/T)*(1+15*T/(8*m))#*(kb**(3./2.)*c**(1./2.)*G**(-1./2.)*hbar**(-5./2.))

def nonrel_energydensitymu(g, m, T, mu): #in eV temp and mass, and eV for mu
    T=T*ev2K #in K now
    m=m*ev #in kg now
    mu=mu*e # in J now
    if (-m*c**2+mu)>0:
        arg=0
    else:
        arg=-m*c**2+mu
    return g*(m*c**2/l_pl**3)*(m*c**2*kb*T/(2*np.pi*(m_pl*c**2)**2))**(3./2.)*np.exp((arg)/(kb*T))*(1+15*kb*T/(8*m*c**2))

def sahahyd(T): #in ev, returns n_hydrogen
    muh=mup(T)+mue(T/10**6)
    T=T*ev2K
    return 4*(mh*T/(m_pl*t_pl*2*np.pi))**(3./2.)*np.exp((muh*ev*10**6-mh)/(kb*T/c**2))/l_pl**3
#    return l_pl**3*n**2*(mh*2*np.pi*m_pl**2*c**2/(me*mp*kb*t*ev2K))**(3./2.)*np.exp(13.6/t)

earlytemp=np.flip(earlytemp)
eps_rel=np.zeros(len(earlytemp))
s_rel=np.zeros(len(earlytemp))
early_a=np.zeros(len(earlytemp))
early_a[-1]=a[decoupling_index]*(t_cmb[decoupling_index]/ev2K)*gstarS(t_cmb[decoupling_index]/ev2K)**(1./3.)/(earlytemp[-1]*gstarS(earlytemp[-1])**(1./3.))
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

backwardsindex_fromdecoupling=np.flip(np.arange(0, early_length))

a_qcd=0
a_nudec=0 #Scale factor at neutrino decoupling (set by for loop)
a_edec=0 #Scale factor at electron positron annihilation (set by for loop)

#CHECK

for i in backwardsindex_fromdecoupling:
    t=earlytemp[i]
    if i!=(early_length-1):
        #print('test')
        early_a[i]=early_a[i+1]*earlytemp[i+1]*gstarS(earlytemp[i+1])**(1./3.)/(t*gstarS(t)**(1./3.))
        early_dm[i]=early_dm[i+1]*(early_a[i+1]/early_a[i])**3
        early_l[i]=early_l[i+1]
    else:
        pass

#DM Freeze Out (for M=1 GeV WIMPs, freeze out is 100 MeV, aka less by a factor of 10)
mev100=np.argmin(abs(earlytemp-100*10**6))
early_dm[:mev100]=np.zeros(mev100)

backwardsindex_array=np.arange(0, len(earlytemp))

backwardsindex_fromdecoupling=np.arange(0, early_length)

testswitch1=0

testswitch2=0

for i in backwardsindex_fromdecoupling:
    t=earlytemp[i]
    #print(t/10**6)
    eps_rel[i]=np.pi**2*gstar(t)*(kb*t*ev2K)**4/(30*(m_pl*c**2*l_pl)**3)
    s_rel[i]=(2*np.pi**2*gstarS(t)*(t*ev2K)**3/45)*kb**4/(c**3*hbar**3)
    if t>qcdphase: #Pre QCD phase transition, early universe
        for particle in standardmodel_preqcd:
            if (t/6.)<particle[1]: #then particle is non-rel/decoupled, has its own array
                af=early_a[-1]*earlytemp[-1]*gstarS(earlytemp[-1])**(1./3.)/(particle[1]*6*gstarS(particle[1]*6)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
    elif t<qcdphase and t>nu_dec: #Post confinement, pre weak decoupling
        if a_qcd==0:
            a_qcd=early_a[i]
        else:
            pass
        for particle in standardmodel_postqcd:
            if (t/6.)<particle[1] and particle[0]!='Neutrinos' and particle[0]!='Electron': #then particle has its own array
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*gstarS(earlytemp[decoupling_index-1])**(1./3.)/(particle[1]*6*gstarS(particle[1]*6)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Electron':
                particle_t=t
                energy=7*np.pi**2*particle[3]*(particle_t*kb*ev2K)**4/(120*(m_pl*c**2*l_pl)**3)*(1+30*(mue(particle_t/10**6)*10**6/particle_t)**2/(7*np.pi**2))
                entropy=0
                #entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
        early_proton[i]=nonrel_energydensitymu(4, mp/ev, t, mup(t)*10**6) #Check degeneracy later
        early_neutron[i]=nonrel_energydensitymu(4, mn/ev, t, mun(t)*10**6)
        early_hydrogen[i]=mh*sahahyd(t)
        early_mup[i]=mup(t)
        early_mun[i]=mun(t)
        protontemp[i]=t
        neutrontemp[i]=t
        early_s_proton[i]=early_proton[i]/protontemp[i]
        early_s_neutron[i]=early_neutron[i]/neutrontemp[i]
    elif t<nu_dec and t>e_dec:
        if a_nudec==0:
            a_nudec=early_a[i]
            protaf=early_a[i]
            protfreeze=t
        else:
            pass
        for particle in standardmodel_postqcd:
            if (t/6.)<particle[1] and particle[0]!='Neutrinos' and particle[0]!='Electron': #then nonrelativistic particle has its own array and isn't decoupled neutrinos
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*gstarS(earlytemp[decoupling_index-1])**(1./3.)/(particle[1]*6*gstarS(particle[1]*6)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=nonrel_energydensity(particle[3], particle[1], particle_t)
                entrop=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Neutrinos':
                particle_t=nu_dec*(a_nudec/early_a[i])
                energy=7*np.pi**2*particle[3]*(particle_t*kb*ev2K)**4/(120*(m_pl*c**2*l_pl)**3)
                entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
                # print("Calculating a neutrino point")
                # print(particle_t)
                # print(i)
            elif particle[0]=='Electron':
                particle_t=t
                energy=7*np.pi**2*particle[3]*(particle_t*kb*ev2K)**4/(120*(m_pl*c**2*l_pl)**3)*(1+30*(mue(particle_t/10**6)*10**6/particle_t)**2/(7*np.pi**2))
                entropy=0
                #entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            else: #particle is relativistic
                pass
        protontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        neutrontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        early_proton[i]=nonrel_energydensitymu(4, mp/ev, protontemp[i], mup(protontemp[i])*10**6) #Check degeneracy later
        early_neutron[i]=nonrel_energydensitymu(4, mn/ev, neutrontemp[i], mun(neutrontemp[i])*10**6)
        early_hydrogen[i]=sahahyd(t)*mh
        early_s_proton[i]=early_proton[i]/protontemp[i]
        early_s_neutron[i]=early_neutron[i]/neutrontemp[i]
    # elif t<e_dec and t>emphase:
        
    else:
        if a_edec==0:
            a_edec=early_a[i]
        else:
            pass
        for particle in standardmodel_postqcd:
            if particle[0]=='Electron': #then nonrelativistic particle has its own array and isn't decoupled neutrinos
                particle_t=t#e_dec*(a_edec/early_a[i])**2
                energy=nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Photon':
                #particle_t=nu_dec*(a_nudec/early_a[i])*(11./4)**(1./3)
                particle_t=(t_cmb[0]/ev2K)*(a[0]/early_a[i])
                energy=np.pi**2*particle[3]*(kb*ev2K*particle_t)**4/(30*(m_pl*c**2*l_pl)**3)
                #np.pi**2*gstar(t)*(kb*t*ev2K)**4/(30*(m_pl*c**2*l_pl)**3)
                entropy=2*np.pi**2*kb**4*2*(particle_t*ev2K)**3/(45*c**3*hbar**3)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
            elif particle[0]=='Neutrinos':
                particle_t=nu_dec*(a_nudec/early_a[i])
                energy=7*np.pi**2*particle[3]*(particle_t*kb*ev2K)**4/(120*(m_pl*c**2*l_pl)**3)
                entropy=2*np.pi**2*kb**4*6*7*(particle_t*ev2K)**3/(45*c**3*hbar**3*8)
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
                # particle_t=particle[1]*6*(af/early_a[i])
                # energy=7*np.pi**2*particle[3]*(kb*ev2K*particle_t)**4/(120*(m_pl*c**2*l_pl)**3)
                # particle[4][i]=energy
            else: #particle is nonrel and not a photon, neutrino, or electron
                af=early_a[decoupling_index-1]*earlytemp[decoupling_index-1]*gstarS(earlytemp[decoupling_index-1])**(1./3.)/(particle[1]*6*gstarS(particle[1]*6)**(1./3.))
                particle_t=particle[1]*6*(af/early_a[i])**2 #Check nonrel_energydens for units
                energy=nonrel_energydensity(particle[3], particle[1], particle_t)
                entropy=energy/particle_t
                particle[4][0][i]=energy
                particle[4][1][i]=particle_t
                particle[4][2][i]=entropy
        protontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        neutrontemp[i]=t#protfreeze*(protaf/early_a[i])**2
        early_proton[i]=nonrel_energydensitymu(4, mp/ev, protontemp[i], mup(protontemp[i])*10**6) #Check degeneracy later
        early_neutron[i]=nonrel_energydensitymu(4, mn/ev, neutrontemp[i], mun(neutrontemp[i])*10**6)
        early_hydrogen[i]=sahahyd(t)*mh
        early_s_proton[i]=early_proton[i]/protontemp[i]
        early_s_neutron[i]=early_neutron[i]/neutrontemp[i]
        

#For hydrogen reionization, use analytic formulas from Mukhanov or other textbooks

#Energy Density
total_bm=standardmodel_postqcd[7][4][0]+standardmodel_postqcd[8][4][0]
for i in range(len(standardmodel_postqcd)):
    if standardmodel_postqcd[i][2] ==True and standardmodel_postqcd[i][0]!='Neutrinos':
        total_bm+=standardmodel_postqcd[i][4][0]
    else:
        pass

for i in range(len(standardmodel_preqcd)):
    if standardmodel_preqcd[i][2] ==True and standardmodel_preqcd[i][0]!='Neutrinos':
        total_bm+=standardmodel_preqcd[i][4][0]
    else:
        pass
    
#Add protons/neutrons
total_bm+=early_proton+early_neutron+early_hydrogen
    
#Entropy Density
for i in range(len(standardmodel_postqcd)):
    if standardmodel_postqcd[i][0]!='Photon' and standardmodel_postqcd[i][0]!='Neutrinos':
        early_s_baryon+=standardmodel_postqcd[i][4][2]
    else:
        pass

for i in range(len(standardmodel_preqcd)):
    if standardmodel_preqcd[i][0]!='Photon' and standardmodel_preqcd[i][0]!='Neutrinos':
        early_s_baryon+=standardmodel_preqcd[i][4][2]
    else:
        pass

total_bm=total_bm+early_proton+early_neutron
total_m=total_bm+early_dm
zero_crossings = np.where(np.diff(np.sign(eps_rel-total_m)))[0]
#zero_crossings = np.where(np.diff(np.sign(standardmodel_preqcd[13][4][0]+standardmodel_postqcd[-1][4][0]-total_m)))[0]

early_zeq=1/early_a[zero_crossings[-1]]-1
theoretical_zeq=(om_bm0+om_dm0)/(om_cmb+om_nu)


#Time/Hubble parameter
#SET PROPER TIME OF CMB, DOES NOT CORRELATE TO MODERN UNIVERSE CODE
set_t=time[0] #set_t=380000*yr
early_eta=np.zeros(early_length)
early_eta[-1]=eta[decoupling_index]
early_t=np.zeros(early_length)
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
early_r_ceh=np.zeros(early_length)
early_volumeceh=np.zeros(early_length)
early_s_ceh=np.zeros(early_length)

backwardsindex_array=np.flip(np.arange(0, len(earlytemp)))

backwardsindex_fromdecoupling=np.flip(np.arange(0, early_length))

for i in backwardsindex_fromdecoupling:
        epstot=total_m[i]+early_l[i]+eps_rel[i]
        early_hub[i]=np.sqrt(8*np.pi*G*epstot/(3*c**2))*mpc/1000 #Assume H=adot/a
        adot[i]=early_hub[i]*early_a[i]
        if i!=backwardsindex_fromdecoupling[0] and eps_rel[i]>total_m[i]:
            early_t[i]=early_t[i+1]*(early_a[i]/early_a[i+1])**2
            early_eta[i]=early_eta[i+1]*(early_a[i]/early_a[i+1])
        elif i!=backwardsindex_fromdecoupling[0] and eps_rel[i]<total_m[i]:
            early_t[i]=(early_t[i+1]**(2./3.)*early_a[i]/early_a[i+1])**(3./2.)
            early_eta[i]=early_eta[i+1]*(early_a[i]/early_a[i+1])**2
        else:
            pass
        early_r_ceh[i]=early_a[i]*(etainf-(early_eta[i]))*c
        early_volumeceh[i]=4*np.pi*early_r_ceh[i]**3/3
        early_s_ceh[i]=(early_r_ceh[i]**2*np.pi*kb*c**3/(G*hbar))/early_volumeceh[i]
        
#%%

#PBHs

def lifetime(m): #m in kg
    m=m*1000/1e10
    return 407*m**3

def betaprime2f(betaprime):
    gam=0.2
    g_pbh=gstar(earlytemp[mev100])
    h_pbh=1#early_hub[mev100]/100
    etot=total_m[mev100]+early_l[mev100]+eps_rel[mev100]
    edm=early_dm[mev100]
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
        sol=(t0/407)**(1/3)*1e10/1000
        frac=1e-7 #Approximate due to evaporation constraints from CMB, EGB, GGB
    return sol*msol, frac #Black hole mass in kg, mass fraction of CDM

def dmdt(m): #dm10/dt
    f=1 #Normalized number of emitted particle species for black hole of mass M (ranges from 1 to 15.35)
    return -5.34*10**(-5)*f/m**2 #m in m10

#m=m10*10^10g
#10^15g-10^17g, 10^14g-10^15g

def fnum(m): #m in msol
    if m*msol*1000>1e17: #only emits photons
        sol=1
    elif m*msol*1000>5e14: #emits electrons and positons as well
        sol=1.569
    else:
        m=hbar*c**3/(8*np.pi*G*kb*m) #convert to temperature
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
    return hbar*c**3/(8*np.pi*G*m*kb)

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
    return 16*np.pi*G*kb*(m0**2-m**2)/(3*hbar*c)


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
    n=eps*massfunc(option)[1]/(massfunc(option)[0]*c**2)
    return n*4*np.pi*kb*G*massfunc(option)[0]**2/(c*hbar) #returns entropy density of PBHs

def spbh(eps, option, m): #eps is CDM energy density, option[0] is BH mass, n is number DENSITY of BHs
    n=np.array(eps*massfunc(option)[1]/(massfunc(option)[0]*c**2))
    return n*4*np.pi*kb*G*(np.array(m))**2/(c*hbar), n #returns entropy density of PBHs


totaldm=np.concatenate((early_dm, eps_dm))
totala=np.concatenate((early_a, a))

pbhtype=0

mass_range0=np.flip(mass_range0)
mass0=np.interp(totaltime[mev100:], time0, mass_range0)
mass0=np.concatenate((np.zeros(len(totaltime)-len(mass0)),np.array(mass0)))

early_pbh0=np.zeros(early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh0[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass0[mev100:])[0]
early_pbh0_noevap=np.zeros(early_length+l)
early_pbh0_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass0[-1])[0]
hawking0=np.zeros(early_length+l)
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

early_wimp0=early_dm*(1-massfunc(pbhtype)[1])



pbhtype=1

mass_range1=np.flip(mass_range1)
mass1=np.interp(totaltime[mev100:], time1, mass_range1)
mass1=np.concatenate((np.zeros(len(totaltime)-len(mass1)),np.array(mass1)))

early_pbh1=np.zeros(early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh1[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass1[mev100:])[0]
early_pbh1_noevap=np.zeros(early_length+l)
early_pbh1_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass1[-1])[0]
#early_wimp1=early_dm*(1-massfunc(pbhtype)[1])
hawking1=np.zeros(early_length+l)
hawking1[(mev100):]=s_hawking(mass1[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass1[mev100:])[1]

pbhtype=2

mass_range2=np.flip(mass_range2)
mass2=np.interp(totaltime[mev100:], time2, mass_range2)
mass2=np.concatenate((np.zeros(len(totaltime)-len(mass2)),np.array(mass2)))

early_pbh2=np.zeros(early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh2[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass2[mev100:])[0]
early_pbh2_noevap=np.zeros(early_length+l)
early_pbh2_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, mass2[-1])[0]
#early_wimp2=early_dm*(1-massfunc(pbhtype)[1])
hawking2=np.zeros(early_length+l)
hawking2[(mev100):]=s_hawking(mass2[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass2[mev100:])[1]

pbhtype=3

mass_range3=np.flip(mass_range3)
mass3=np.interp(totaltime[mev100:], time3, mass_range3)
mass3=np.concatenate((np.zeros(len(totaltime)-len(mass3)),np.array(mass3)))

early_pbh3=np.zeros(early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh3[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass3[mev100:])[0]
early_pbh3_noevap=np.zeros(early_length+l)
early_pbh3_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, massfunc(3)[0])[0]
hawking3=np.zeros(early_length+l)
hawking3[(mev100):]=s_hawking(mass3[mev100:], massfunc(pbhtype)[0])*spbh(totaldm[mev100:], pbhtype, mass3[mev100:])[1]

figpbh3, axpbh3 = plt.subplots(1,1)
axpbh3.scatter(np.log10(totala), np.log10(early_pbh3), c='c', marker='x')
axpbh3.scatter(np.log10(totala), np.log10(early_pbh3_noevap), c='y', marker='+')

pbhtype=4

mass_range4=np.flip(mass_range4)
mass4=np.interp(totaltime[mev100:], time4, mass_range4)

mass4=np.concatenate((np.zeros(len(totaltime)-len(mass4)),np.array(mass4)))

early_pbh4=np.zeros(early_length+l)
#early_pbh0[mev100]=spbh0(early_dm[mev100], pbhtype) #Assumes PBH mass function assumes mass fraction of DM at time of PBH creation
early_pbh4[(mev100):]=spbh(totaldm[mev100:], pbhtype, mass4[mev100:])[0]
early_pbh4_noevap=np.zeros(early_length+l)
early_pbh4_noevap[mev100:]=spbh(totaldm[mev100:], pbhtype, massfunc(4)[0])[0]
hawking4=np.zeros(early_length+l)
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

fig, ax = plt.subplots(1,1)
# ax.plot(np.log10(early_a),np.log10(standardmodel_preqcd[0][4][1]+standardmodel_postqcd[0][4][1]), c='#0EAD69', label='Top')
# ax.plot(np.log10(early_a),np.log10(standardmodel_preqcd[1][4][1]+standardmodel_postqcd[4][4][1]), c='#EE4266', label="Bottom")
# ax.plot(np.log10(early_a),np.log10(standardmodel_preqcd[2][4][1]+standardmodel_postqcd[6][4][1]), c='#FFD23F', label='Charm')
# ax.plot(np.log10(early_a),np.log10(standardmodel_preqcd[3][4][1]),c='#50E1BF', label="Strange")
# ax.plot(np.log10(early_a),np.log10(standardmodel_preqcd[4][4][1]), c='#8B1EB3', label="Down")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[5][4][1]), c='#000000', label="Up")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[6][4][1]), label="Gluon")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[7][4][1]+standardmodel_postqcd[5][4][1]), label="Tau")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[8][4][1]+standardmodel_postqcd[9][4][1]), label="Muon")
ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[9][4][0]+standardmodel_postqcd[-2][4][0]), label="Electron")
ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[10][4][0]+standardmodel_postqcd[-3][4][0]), label="Neutrinos")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[11][4][1]+standardmodel_postqcd[3][4][1]), label="W Boson")
# ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[12][4][1]+standardmodel_postqcd[2][4][1]), label="Z Boson")
ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[13][4][0]+standardmodel_postqcd[-1][4][0]), label="Photon")
#ax.plot(np.log10(early_a), np.log10(standardmodel_preqcd[14][4][1]+standardmodel_postqcd[1][4][1]), label="Higgs")
ax.plot(np.log10(early_a), np.log10(standardmodel_postqcd[-6][4][0]), label="Charged Pions")
ax.plot(np.log10(early_a), np.log10(standardmodel_postqcd[-5][4][0]), label="Pi0")
ax.plot(np.log10(early_a), np.log10(early_proton), label='Protons')
ax.plot(np.log10(early_a), np.log10(early_neutron), label='Neutrons')
ax.plot(np.log10(early_a), np.log10(early_dm), label='Dark Matter')
ax.plot(np.log10(early_a), np.log10(eps_rel), label='Relativistic')
ax.plot(np.log10(early_a), np.log10(total_bm), linestyle='dotted', label="All Fermions")
ax.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
ax.set_xlabel("Scale Factor")
ax.set_ylabel(r'Energy Density J/m$^3$')
ax.set_ylim(-25, 50)
ax.set_title("Log-Log Energy Density vs Scale Factor")
ax.legend()
fig.set_size_inches(9, 5)
# fig.savefig('early.eps', bbox_inches='tight')

figa, axa = plt.subplots(1,1)  
axa.plot(np.log10(early_a), np.log10(eps_rel), label='Relativistic')
axa.plot(np.log10(early_a), np.log10(total_m), label="All Matter")
axa.axvline(x=np.log10(1/(1+zeq)), linestyle='dotted', label='Matter-Radiation Equality')
axa.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
axa.set_xlabel("Scale Factor")
axa.set_ylabel(r'Energy Density J/m$^3$')
#axa.set_ylim(0, 2.33)
#axa.set_xlim(-3.9, -3.4)
axa.set_title("Matter versus Radiation")
axa.legend()
figa.set_size_inches(9, 5)
figa.savefig('mattervradiation.eps', bbox_inches='tight')

figc, axc = plt.subplots(1,1)  
axc.plot(np.log10(early_a), np.log10(eps_rel), label='Relativistic')
axc.plot(np.log10(early_a), np.log10(standardmodel_preqcd[13][4][0]+standardmodel_postqcd[-1][4][0]), label="Photon")
axc.plot(np.log10(early_a), np.log10(total_m), label="All Matter")
axc.axvline(x=np.log10(1/(1+zeq)), linestyle='dotted', label='Matter-Radiation Equality')
axc.set_xlabel("Scale Factor")
axc.set_ylabel(r'Energy Density J/m$^3$')
axc.set_ylim(-2.5, 5)
axc.set_xlim(-4.25, -2.75)
axc.set_title("Matter-Radiation Equality Check")
axc.legend()
figc.set_size_inches(9, 5)
figc.savefig('earlycheck.eps', bbox_inches='tight')

figd, axd = plt.subplots(1,1)  
axd.plot(np.log10(early_a), np.log10(standardmodel_preqcd[-6][4][0]/me+standardmodel_postqcd[-2][4][0]/me), label="Electrons")
axd.plot(np.log10(early_a), np.log10(early_proton/mp), label="Protons")
#axd.axvline(x=np.log10(1/(1+zeq)), linestyle='dotted', label='Matter-Radiation Equality')
axd.set_xlabel("Scale Factor")
axd.set_ylabel(r'Number Density #/m$^3$')
axd.set_ylim(0, 75)
#axd.set_xlim(-4.25, -2.75)
axd.set_title("Charge Neutrality Check")
axd.legend()
figd.set_size_inches(9, 5)
figd.savefig('earlychargeneutrality.eps', bbox_inches='tight')

figs, axs = plt.subplots(1,1)  
axs.plot(np.log10(early_a), np.log10(s_rel), label='Relativistic')
# axs.plot(np.log10(early_a),np.log10(standardmodel_preqcd[0][4][1]+standardmodel_postqcd[0][4][1]), c='#0EAD69', label='Top')
# axs.plot(np.log10(early_a),np.log10(standardmodel_preqcd[1][4][1]+standardmodel_postqcd[4][4][1]), c='#EE4266', label="Bottom")
# axs.plot(np.log10(early_a),np.log10(standardmodel_preqcd[2][4][1]+standardmodel_postqcd[6][4][1]), c='#FFD23F', label='Charm')
# axs.plot(np.log10(early_a),np.log10(standardmodel_preqcd[3][4][1]),c='#50E1BF', label="Strange")
# axs.plot(np.log10(early_a),np.log10(standardmodel_preqcd[4][4][1]), c='#8B1EB3', label="Down")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[5][4][1]), c='#000000', label="Up")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[6][4][1]), label="Gluon")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[7][4][1]+standardmodel_postqcd[5][4][1]), label="Tau")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[8][4][1]+standardmodel_postqcd[9][4][1]), label="Muon")
#axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[9][4][2]+standardmodel_postqcd[-2][4][2]), label="Electron")
axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[10][4][2]+standardmodel_postqcd[-3][4][2]), label="Neutrinos")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[11][4][1]+standardmodel_postqcd[3][4][1]), label="W Boson")
# axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[12][4][1]+standardmodel_postqcd[2][4][1]), label="Z Boson")
axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[13][4][2]+standardmodel_postqcd[-1][4][2]), label="Photon")
axs.fill_between(np.log10(early_a), np.log10(early_pbh0[:1000]), np.log10(early_pbh1[:1000]), label="PBHs (f=1)",  facecolor="#DCF6FF")
axs.plot(np.log10(early_a), np.log10(early_pbh2[:1000]), label="Largest PBH (f=0.3)")
axs.plot(np.log10(early_a), np.log10(early_pbh4[:1000]), label="Evaporated PBH example")
axs.plot(np.log10(early_a), np.log10(hawking4[:1000]), label="Evaporated PBH radiation")
#axs.plot(np.log10(early_a), np.log10(standardmodel_preqcd[14][4][1]+standardmodel_postqcd[1][4][1]), label="Higgs")
#axs.plot(np.log10(early_a), np.log10(standardmodel_postqcd[-6][4][2]), label="Charged Pions")
#axs.plot(np.log10(early_a), np.log10(standardmodel_postqcd[-5][4][2]), label="Pi0")
#axs.plot(np.log10(early_a), np.log10(early_s_proton), label='Protons')
#axs.plot(np.log10(early_a), np.log10(early_s_neutron), label='Neutrons')
axs.plot(np.log10(early_a), np.log10(early_s_baryon+early_s_proton+early_s_neutron), linestyle='dotted', label="Baryons")
axs.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
axs.set_xlabel("Scale Factor")
axs.set_ylabel(r'Entropy Density')
axs.set_ylim(-25, 50)
axs.set_title("Log-Log Entropy Density vs Scale Factor")
axs.legend()
figs.set_size_inches(9, 5)
figs.savefig('earlys.eps', bbox_inches='tight')

figb, axb, = plt.subplots(1,1)
axb.plot(np.log10(early_a), np.log10(early_proton), label='Protons')
axb.plot(np.log10(early_a), np.log10(early_neutron), linestyle='dashed', label='Neutrons')
axb.plot(np.log10(early_a), np.log10(early_hydrogen), label='Hydrogen')
axb.plot(np.log10(early_a), np.log10(standardmodel_preqcd[-2][4][0]+standardmodel_postqcd[-2][4][0]), label='Electrons')
#axb.plot(np.log10(early_a), np.log10(total_bm), label="All Fermions (-p and n)")
axb.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
axb.axvline(x=np.log10(a_edec), c='r', linestyle='dotted', label='Electron Decoupling')
axb.axvline(x=np.log10(early_a[943]), c='r', linestyle='dotted', label='Recombination')
axb.set_ylim(-10,40)
axb.set_xlabel("Scale Factor")
axb.set_ylabel(r'Energy Density J/m$^3$')
axb.set_title("Log-Log Energy Density vs Scale Factor")
axb.legend()
figb.savefig('earlybm.eps', bbox_inches='tight')

# figr, axr, = plt.subplots(1,1)
# axr.plot(np.log10(early_a), np.log10(npratio(earlytemp/10**6)), label='Theoretical Ratio')
# axr.plot(np.log10(early_a), np.log10((early_neutron/mn)/(early_proton/mp)), label='Calculated Ratio from Code')
# axr.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
# axr.set_xlabel("Scale Factor")
# axr.set_ylabel(r'$n_n/n_p$')
# axr.set_xlim(-13,-9.4)
# axr.set_ylim(-0.5,0.01)
# axr.set_title("Log-Log Neutron to Proton Ration vs Scale Factor")
# axr.legend()
# figr.savefig('earlyratio.eps', bbox_inches='tight')

# figmu, axmu, = plt.subplots(1,1)
# axmu.plot(np.log10(earlytemp), np.log10(early_mup), label=r'$\mu_p$')
# axmu.plot(np.log10(earlytemp), np.log10(early_mun), label=r'$\mu_n$')
# #axmu.axvline(x=np.log10(a_qcd), c='#0EAD69', linestyle='dotted', label='QCD')
# axmu.set_xlabel("Temperature [MeV]")
# axmu.set_ylabel(r'$\mu$ [MeV]')
# # axmu.set_xlim(-13,-9.4)
# # axmu.set_ylim(-0.5,0.01)
# axmu.set_title("Log-Log $\mu$ vs Temperature")
# axmu.legend()
# figmu.savefig('earlymu.eps', bbox_inches='tight')


# figpbh, axpbh = plt.subplots(1,1)
# #axpbh.fill_between(np.log10(totala), np.log10(early_pbh0), np.log10(early_pbh1), label="PBHs (f=1)",  facecolor="#DCF6FF")
# axpbh.plot(np.log10(totala), np.log10(early_pbh2), label="Largest PBH (f=0.3)")
# axpbh.plot(np.log10(totala), np.log10(early_pbh4), label="Evaporated PBH example")
# axpbh.plot(np.log10(totala), np.log10(hawking4), label="Evaporated PBH radiation")
# axpbh.set_xlabel("Scale Factor")
# axpbh.set_ylabel(r'Entropy Density')
# axpbh.set_ylim(-25, 50)
# axpbh.set_title("Log-Log PBH Entropy Density vs Scale Factor")
# axpbh.legend()
# figpbh.set_size_inches(9, 5)
# figpbh.savefig('pbhevap.eps', bbox_inches='tight')

#Entropy Densities in Early Universe

print("Energy at Decoupling:  Early Code, Late Code")
print("-------------------")
print("Cosmic Microwave Background: %10.2E, \t(%10.2E)" % (standardmodel_postqcd[-1][4][0][-1], eps_cmb[0]))
print("CMB Temp                   : %10.2E, \t(%10.2E)" % (standardmodel_postqcd[-1][4][1][-1]*ev2K, t_cmb[0]))
print("Primordial Neutrinos       : %10.2E \t(%10.2E)" % (standardmodel_postqcd[-3][4][0][-1], eps_nu[0]))
print("Neutrino Temp              : %10.2E \t(%10.2E)" % (standardmodel_postqcd[-3][4][1][-1]*ev2K, t_nu[0]))
print("Dark Matter                : %10.2E \t(%10.2E)" % (total_m[-1], eps_dm[0]))



#%%

#Significant features for plots
strpeak_arg=np.argmax(abs(strdot(hub)))
lambdadom_arg=np.argmin(abs(eps-2*eps_l))
zeq_arg=np.argmin(abs(a-1/(1+zeq)))
zdec_arg=np.argmin(abs(a-1/(1+zdec)))
zrec_arg=np.argmin(abs(a-1/(1+zrec)))

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
    return zigzag(lda)*(m/msol)*lda*Leddsol*(1-etarad(mdt))/(etarad(mdt)*c**2)

def mdot(m):
    Medd=1.3*10**(38)*m/(0.1*msol*c**2)
    # mdt0=1
    rhsmdot0=zigzagint*delta*(m/msol)*Leddsol*(1-etarad0)/(etarad0*c**2)
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

#%%
"""
Plots for energy density, temperature, entropy density
"""

#Energy Density    
fig1, ax1 = plt.subplots(1,1)  
fig1.set_size_inches(18.5, 10.5)
ax1.plot(np.log10(a),np.log10(eps_bm), c='#0EAD69', label='Baryonic Matter')
ax1.plot(np.log10(a),np.log10(eps_cmb), c='#EE4266', label="CMB")
ax1.plot(np.log10(a),np.log10(eps_nu), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
ax1.plot(np.log10(a),np.log10(eps_str),c='#50E1BF', label="Stars")
ax1.plot(np.log10(a),np.log10(eps_dm), c='#8B1EB3', label="Dark Matter")
ax1.plot(np.log10(a), np.log10(eps_l), c='#000000', label="Lambda")
ax1.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
ax1.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
ax1.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax1.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax1.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax1.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax1.set_xlabel("Scale Factor")
ax1.set_ylabel(r'Energy Density J/m$^3$')
ax1.set_title("Log-Log Energy Density vs Scale Factor")
ax1.legend()
fig1.savefig('energydensity.eps')  

#Temperature
fig2, ax2 = plt.subplots(1,1)    
fig2.set_size_inches(18.5, 10.5)
ax2.plot(np.log10(1/a-1),np.log10(t_bm), c='#0EAD69', label='Baryonic Matter')
ax2.plot(np.log10(1/a-1),np.log10(t_cmb), c='#EE4266', label="CMB")
ax2.plot(np.log10(1/a-1),np.log10(t_nu), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
ax2.axvline(x=np.log10(zeq), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
ax2.axvline(x=np.log10(zrec), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
ax2.axvline(x=np.log10(zdec), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax2.axvline(x=np.log10(1/a[z30]-1), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax2.axvline(x=np.log10(1/a[strpeak_arg]-1), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax2.axvline(x=np.log10(1/a[lambdadom_arg]-1), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax2.set_xlabel("Redshift")
ax2.set_ylabel("Temperature (K)")
ax2.set_title("Log-Log Temperature vs Scale Factor")
ax2.legend()
fig2.savefig('temperaturez.eps')

#Entropy Density
fig3, ax3 = plt.subplots(1,1) 
fig3.set_size_inches(18.5, 10.5)   
ax3.plot(np.log10(a),np.log10(s_bm), c='#0EAD69', label=r'Baryonic Matter')
ax3.plot(np.log10(a),np.log10(s_cmb), c='#EE4266', linestyle='dashed', label=r'CMB')
ax3.plot(np.log10(a),np.log10(s_nu), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
#ax3.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax3.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax3.plot(np.log10(a),np.log10(s_ceh), c='#000000', label=r'CEH')
ax3.fill_between(np.log10(a), np.log10(s_ceh_h), np.log10(s_ceh_l), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
ax3.plot(np.log10(a),np.log10(s_smbh),c='#8B1EB3', label=r'Supermassive Black Holes')
ax3.fill_between(np.log10(a), np.log10(s_smbhlow), np.log10(s_smbhhigh), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
ax3.plot(np.log10(a),np.log10(s_bh),c='#50E1BF', label=r'Stellar Mass Black Holes')
#ax3.fill_between(np.log10(a), np.log10(s_bhlow), np.log10(s_bhhigh), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
ax3.fill_between(np.log10(totala[1000:]), np.log10(early_pbh0[1000:]), np.log10(early_pbh1[1000:]), label="PBHs (f=1)",  facecolor="#DCF6FF")
ax3.plot(np.log10(totala[1000:]), np.log10(early_pbh2[1000:]), label="Largest PBH (f=0.3)")
ax3.plot(np.log10(totala[1000:]), np.log10(early_pbh4[1000:]), label=r"Evaporating PBH ($M=5\times 10^{-20} M_{\odot}$)")
ax3.plot(np.log10(totala[1000:]), np.log10(hawking4[1000:]), label="Hawking Radiation", c='#FF00FF', linestyle='dashed')
#ax3.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
#ax3.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
#ax3.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax3.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax3.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax3.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax3.set_xlabel('Scale Factor')
ax3.set_ylabel('Entropy Density')
ax3.set_title(r'Log-Log Entropy Density vs Scale Factor')
ax3.set_ylim(bottom=-22)
ax3.legend(loc=2)
fig3.savefig("s_evolution.eps")
#fig3.savefig("s_evolution_noeagle.eps")

#Entropy Total
fig4, ax4 = plt.subplots(1,1)
fig4.set_size_inches(18.5, 10.5)
ax4.plot(np.log10(a),np.log10(volumeceh*s_bm), c='#0EAD69', label='Baryonic Matter')
ax4.plot(np.log10(a),np.log10(volumeceh*s_cmb), c='#EE4266', label=r'CMB')
ax4.plot(np.log10(a),np.log10(volumeceh*s_nu), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
#ax4.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax4.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax4.plot(np.log10(a),np.log10(stot_ceh), c='#000000', label=r'CEH')
ax4.plot(np.log10(a),np.log10(volumeceh*s_smbh),c='#8B1EB3', label=r'Supermassive Black Holes')
ax4.plot(np.log10(a),np.log10(volumeceh*s_bh),c='#50E1BF', label=r'Stellar Mass Black Holes')
ax4.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
ax4.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
ax4.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax4.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax4.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax4.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax4.set_xlabel('Scale Factor')
ax4.set_ylabel('Entropy')
ax4.set_title(r'Log-Log Entropy vs Scale Factor')
ax4.set_ylim(bottom=50)
ax4.legend(loc=2)
fig4.savefig('totalentropy.eps')

#Cosmic Event Horizon
fig5, ax5 = plt.subplots(1,1)    
fig5.set_size_inches(18.5, 10.5)
ax5.plot(np.log10(a),np.log10(r_ceh), c='#8B1EB3', label="CEH")
ax5.fill_between(np.log10(a), np.log10(r_ceh_l), np.log10(r_ceh_h), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
ax5.plot(np.log10(a),np.log10(r_p), c='#50E1BF', label="PH")
ax5.set_xlabel("log(a)")
ax5.set_ylabel("log(R_CEH)")
ax5.legend()
ax5.set_title("Cosmic Event Horizon Radius vs Scale Factor")
fig5.savefig('horizons.eps')

# inflection=0
# for i in range(len(a)-2):
#     agrad1=(a[i+1]-a[i])/(t[i+1]-t[i])
#     agrad2=(a[i+2]-a[i+1])/(t[i+2]-t[i+1])
#     if (agrad2>agrad1)==True:
#         inflection=i+1
#         pass

# #Time vs Scale Factor
# fig6, ax6 = plt.subplots(1,1)    
# fig6.set_size_inches(18.5, 10.5)
# ax6.plot(((t/(10**9*yr))),(a))
# #ax6.axvline(x=t[zeq_arg]/(10**9*yr), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# #ax6.axvline(x=t[zrec_arg]/(10**9*yr), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# #ax6.axvline(x=t[zdec_arg]/(10**9*yr), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax6.axvline(x=t[z30]/(10**9*yr), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax6.axvline(x=t[strpeak_arg]/(10**9*yr), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax6.axvline(x=t[lambdadom_arg]/(10**9*yr), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax6.set_xlabel("Proper Time (Gyr)")
# ax6.set_ylabel("Scale Factor")
# ax6.set_title("Scale Factor vs Age")
# ax6.legend()
# fig6.savefig('scalefactor.eps')

# #Entropy (conformal volume)
# fig7, ax7 = plt.subplots(1,1) 
# fig7.set_size_inches(18.5, 10.5)   
# ax7.plot(np.log10(a),np.log10(s_bm*a**3), c='#0EAD69', label=r'Baryonic Matter')
# ax7.plot(np.log10(a),np.log10(s_cmb*a**3), c='#EE4266', linestyle='dashed', label=r'CMB')
# ax7.plot(np.log10(a),np.log10(s_nu*a**3), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
# #ax7.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
# #ax7.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
# ax7.plot(np.log10(a),np.log10(stot_ceh), c='#000000', label=r'CEH')
# #ax7.fill_between(np.log10(a), np.log10(s_ceh_h), np.log10(s_ceh_l), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
# ax7.plot(np.log10(a),np.log10(s_smbh*a**3),c='#8B1EB3', label=r'Supermassive Black Holes')
# #ax7.fill_between(np.log10(a), np.log10(s_smbhlow), np.log10(s_smbhhigh), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
# ax7.plot(np.log10(a),np.log10(s_bh*a**3),c='#50E1BF', label=r'Stellar Mass Black Holes')
# #ax7.fill_between(np.log10(a), np.log10(s_bhlow), np.log10(s_bhhigh), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
# ax7.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax7.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax7.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax7.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax7.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax7.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax7.set_xlabel('Scale Factor')
# ax7.set_ylabel('Entropy Density')
# ax7.set_title(r'Log-Log Conformal Volume Entropy vs Scale Factor')
# ax7.set_ylim(bottom=-26)
# ax7.legend(loc=2)
# fig7.savefig("conformals_evolution.eps")

# #BH Zoom
# fig8, ax8 = plt.subplots(1,1) 
# fig8.set_size_inches(18.5, 10.5)   
# #ax8.plot(np.log10(a[z30:]),np.log10(s_bm[z30:]), c='#0EAD69', label=r'Baryonic Matter')
# ax8.plot(np.log10(a[z30:]),np.log10(s_cmb[z30:]), c='#EE4266', linestyle='dashed', label=r'CMB')
# ax8.plot(np.log10(a[z30:]),np.log10(s_nu[z30:]), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
# #ax3.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
# #ax3.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
# #ax8.plot(np.log10(a),np.log10(s_ceh), c='#000000', label=r'CEH')
# #ax8.fill_between(np.log10(a), np.log10(s_ceh_h), np.log10(s_ceh_l), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
# ax8.plot(np.log10(a[z30:]),np.log10(s_smbh[z30:]),c='#8B1EB3', label=r'Supermassive Black Holes')
# ax8.fill_between(np.log10(a[z30:]), np.log10(s_smbhlow[z30:]), np.log10(s_smbhhigh[z30:]), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
# ax8.plot(np.log10(a[z30:]),np.log10(s_bh[z30:]),c='#50E1BF', label=r'Stellar Mass Black Holes')
# ax8.fill_between(np.log10(a[z30:]), np.log10(s_bhlow[z30:]), np.log10(s_bhhigh[z30:]), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
# ax8.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax8.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax8.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax8.set_xlabel('Scale Factor')
# ax8.set_ylabel('Entropy Density')
# ax8.set_title(r'Log-Log Entropy Density vs Scale Factor')
# ax8.set_ylim(bottom=-17, top=5)
# #ax8.set_xlim(left=a[z30])
# ax8.legend(loc=2)
# fig8.savefig("bh_zoom.eps")

# #Star zoom
# fig9, ax9 = plt.subplots(1,1) 
# fig9.set_size_inches(18.5, 10.5)   
# ax9.plot(np.log10(a[z30:]),np.log10(eps_bm[z30:]), c='#0EAD69', label='Baryonic Matter')
# ax9.plot(np.log10(a[z30:]),np.log10(eps_cmb[z30:]), c='#EE4266', label="CMB")
# ax9.plot(np.log10(a[z30:]),np.log10(eps_nu[z30:]), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
# ax9.plot(np.log10(a[z30:]),np.log10(eps_str[z30:]),c='#50E1BF', label="Stars")
# ax9.plot(np.log10(a[z30:]),np.log10(eps_dm[z30:]), c='#8B1EB3', label="Dark Matter")
# ax9.plot(np.log10(a[z30:]), np.log10(eps_l[z30:]), c='#000000', label="Lambda")
# ax9.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax9.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax9.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax9.set_xlabel("Scale Factor")
# ax9.set_ylabel(r'Energy Density J/m$^3$')
# ax9.set_title("Log-Log Energy Density vs Scale Factor")
# ax9.legend()
# #ax9.set_ylim(bottom=-17, top=5)
# #ax9.set_xlim(left=a[z30])
# fig9.savefig("star_zoom.eps")

# #Entropy (conformal volume)
# stotconformal=(s_bm+s_cmb+s_nu+s_smbh+s_bh)*a**3+stot_ceh
# fig10, ax10 = plt.subplots(1,1) 
# fig10.set_size_inches(18.5, 10.5)   
# ax10.plot(np.log10(a),np.log10(stotconformal))
# ax10.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax10.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax10.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax10.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax10.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax10.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax10.set_xlabel('Scale Factor')
# ax10.set_ylabel('Total Entropy')
# ax10.set_title(r'Log-Log Summed Conformal Volume Entropy vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax10.legend()
# fig10.savefig("summedconformals_evolution.eps")

# #Entropy (CEH volume)
# stotceh=(s_bm+s_cmb+s_nu+s_smbh+s_bh)*volumeceh+stot_ceh
# fig11, ax11 = plt.subplots(1,1) 
# fig11.set_size_inches(18.5, 10.5)   
# ax11.plot(np.log10(a),np.log10(stotceh))
# ax11.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax11.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax11.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax11.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax11.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax11.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax11.set_xlabel('Scale Factor')
# ax11.set_ylabel('Total Entropy')
# ax11.set_title(r'Log-Log Summed CEH Volume Entropy vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax11.legend()
# fig11.savefig("summedcehs_evolution.eps")

# #Entropy Density total
# stot_dens=(s_bm+s_cmb+s_nu+s_smbh+s_bh)+s_ceh
# fig11, ax11 = plt.subplots(1,1) 
# fig11.set_size_inches(18.5, 10.5)   
# ax11.plot(np.log10(a),np.log10(stot_dens))
# ax11.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax11.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax11.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax11.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax11.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax11.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax11.set_xlabel('Scale Factor')
# ax11.set_ylabel('Total Entropy')
# ax11.set_title(r'Log-Log Summed Entropy Density vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax11.legend()
# fig11.savefig("summedsdens_evolution.eps")

# #Entropy Density total minus CEH
# stot_densnoceh=(s_bm+s_cmb+s_nu+s_smbh+s_bh)
# fig12, ax12 = plt.subplots(1,1) 
# fig12.set_size_inches(18.5, 10.5)   
# ax12.plot(np.log10(a),np.log10(stot_densnoceh))
# ax12.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax12.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax12.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax12.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax12.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax12.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax12.set_xlabel('Scale Factor')
# ax12.set_ylabel('Total Entropy')
# ax12.set_title(r'Log-Log Summed Entropy Density (No CEH) vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax12.legend()
# fig12.savefig("summedsdensnoceh_evolution.eps")

# #Entropy in conformal volume minus CEH
# stot_conformalnoceh=(s_bm+s_cmb+s_nu+s_smbh+s_bh)*a**3
# fig13, ax13 = plt.subplots(1,1) 
# fig13.set_size_inches(18.5, 10.5)   
# ax13.plot(np.log10(a),np.log10(stot_conformalnoceh))
# ax13.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax13.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax13.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax13.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax13.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax13.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax13.set_xlabel('Scale Factor')
# ax13.set_ylabel('Total Entropy')
# ax13.set_title(r'Log-Log Summed Entropy (Conformal Volume, No CEH) vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax13.legend()
# fig13.savefig("summedsconformalnoceh_evolution.eps")

# #Entropy in conformal volume minus CEH
# stot_nocehvolceh=(s_bm+s_cmb+s_nu+s_smbh+s_bh)*volumeceh
# fig14, ax14 = plt.subplots(1,1) 
# fig14.set_size_inches(18.5, 10.5)   
# ax14.plot(np.log10(a),np.log10(stot_nocehvolceh))
# ax14.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
# ax14.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
# ax14.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
# ax14.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
# ax14.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax14.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax14.set_xlabel('Scale Factor')
# ax14.set_ylabel('Total Entropy')
# ax14.set_title(r'Log-Log Summed Entropy (CEH Volume, No CEH) vs Scale Factor')
# #ax10.set_ylim(bottom=-26)
# ax14.legend()
# fig14.savefig("summedsnocehvolceh_evolution.eps")

# #s_bm[z20:]=np.interp(a[z20:], np.flip(bma), np.flip(bm_eagleresults))*kb

# fig15, ax15 = plt.subplots(1,1) 
# fig15.set_size_inches(18.5, 10.5)   
# ax15.plot(np.log10(a[(z20):]),np.log10(s_bm[(z20):]), c='#0EAD69', label=r'Baryonic Matter')
# #ax15.plot(np.log10(a[(z20-10):]),np.log10(s_cmb[(z20-10):]), c='#EE4266', linestyle='dashed', label=r'CMB')
# #ax15.plot(np.log10(a[(z20-10):]),np.log10(s_nu[(z20-10):]), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
# #ax3.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
# #ax3.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
# #ax15.plot(np.log10(a[(z20-10):]),np.log10(s_ceh[(z20-10):]), c='#000000', label=r'CEH')
# #ax15.fill_between(np.log10(a[(z20-10):]), np.log10(s_ceh_h[(z20-10):]), np.log10(s_ceh_l[(z20-10):]), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
# #ax15.plot(np.log10(a[(z20-10):]),np.log10(s_smbh[(z20-10):]),c='#8B1EB3', label=r'Supermassive Black Holes')
# #ax15.fill_between(np.log10(a[(z20-10):]), np.log10(s_smbhlow[(z20-10):]), np.log10(s_smbhhigh[(z20-10):]), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
# #ax15.plot(np.log10(a[(z20-10):]),np.log10(s_bh[(z20-10):]),c='#50E1BF', label=r'Stellar Mass Black Holes')
# #ax15.fill_between(np.log10(a[(z20-10):]), np.log10(s_bhlow[(z20-10):]), np.log10(s_bhhigh[(z20-10):]), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
# ax15.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
# ax15.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
# ax15.set_xlabel('Scale Factor')
# ax15.set_ylabel('Entropy Density')
# ax15.set_title(r'Log-Log Entropy Density vs Scale Factor')
# ax15.set_ylim(bottom=-21)
# ax15.legend(loc=2)
# fig15.savefig("eaglezoom.eps")


#Entropy Density
fig16, ax16 = plt.subplots(1,1) 
fig16.set_size_inches(18.5, 10.5)   
ax16.plot(np.log10(totala),np.log10(np.concatenate((early_s_baryon+early_s_proton+early_s_neutron, s_bm))), c='#0EAD69', label=r'Baryonic Matter')
ax16.plot(np.log10(totala),np.log10(np.concatenate((s_rel, s_cmb))), c='#EE4266', linestyle='dashed', label=r'CMB')
ax16.plot(np.log10(totala),np.log10(np.concatenate((standardmodel_preqcd[10][4][2]+standardmodel_postqcd[-3][4][2],s_nu))), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
#ax16.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax16.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax16.plot(np.log10(totala),np.log10(np.concatenate((early_s_ceh,s_ceh))), c='#000000', label=r'CEH')
ax16.fill_between(np.log10(a), np.log10(s_ceh_h), np.log10(s_ceh_l), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
ax16.plot(np.log10(a),np.log10(s_smbh),c='#8B1EB3', label=r'Supermassive Black Holes')
ax16.fill_between(np.log10(a), np.log10(s_smbhlow), np.log10(s_smbhhigh), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
ax16.plot(np.log10(a),np.log10(s_bh),c='#50E1BF', label=r'Stellar Mass Black Holes')
ax16.fill_between(np.log10(a), np.log10(s_bhlow), np.log10(s_bhhigh), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
ax16.fill_between(np.log10(totala), np.log10(early_pbh0), np.log10(early_pbh1), label="PBHs (f=1)",  facecolor="#DCF6FF")
ax16.plot(np.log10(totala), np.log10(early_pbh2), label="Largest PBH (f=0.3)")
ax16.plot(np.log10(totala), np.log10(early_pbh4), label=r"Evaporating PBH ($M=5\times 10^{-20} M_{\odot}$)")
ax16.plot(np.log10(totala), np.log10(hawking4), label="Hawking Radiation", c='#FF00FF', linestyle='dashed')
ax16.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
ax16.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
ax16.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax16.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax16.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax16.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax16.set_xlabel('Scale Factor')
ax16.set_ylabel('Entropy Density')
ax16.set_title(r'Log-Log Entropy Density vs Scale Factor')
ax16.set_ylim(bottom=-22)
ax16.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
fig16.savefig("s_evolution_full.eps")

#Entropy in a conformal volume, entire time domain
fig17, ax17 = plt.subplots(1,1) 
fig17.set_size_inches(18.5, 10.5)   
ax17.plot(np.log10(totala),np.log10(totala**3*np.concatenate((early_s_baryon+early_s_proton+early_s_neutron, s_bm))), c='#0EAD69', label=r'Baryonic Matter')
ax17.plot(np.log10(totala),np.log10(totala**3*np.concatenate((s_rel, s_cmb))), c='#EE4266', linestyle='dashed', label=r'CMB')
ax17.plot(np.log10(totala),np.log10(totala**3*np.concatenate((standardmodel_preqcd[10][4][2]+standardmodel_postqcd[-3][4][2],s_nu))), c='#FFD23F', label=r'$\nu_{\mathrm{primordial}}$')
#ax17.plot(np.log10(a),np.log10(eps_dm), c='c', label="Dark Matter")
#ax17.plot(np.log10(a), np.log10(eps_l), c='g', label="Lambda")
ax17.plot(np.log10(totala),np.log10(totala**3*np.concatenate((early_s_ceh,s_ceh))), c='#000000', label=r'CEH')
ax17.fill_between(np.log10(a), np.log10(a**3*s_ceh_h), np.log10(a**3*s_ceh_l), edgecolor='#FFFFFF', facecolor='#B3B3B3', linestyle='dashdot')
ax17.plot(np.log10(a),np.log10(a**3*s_smbh),c='#8B1EB3', label=r'Supermassive Black Holes')
ax17.fill_between(np.log10(a), np.log10(a**3*s_smbhlow), np.log10(a**3*s_smbhhigh), edgecolor='#FFFFFF', facecolor='#F2CFFF', linestyle='dashdot')
ax17.plot(np.log10(a),np.log10(a**3*s_bh),c='#50E1BF', label=r'Stellar Mass Black Holes')
ax17.fill_between(np.log10(a), np.log10(a**3*s_bhlow), np.log10(a**3*s_bhhigh), edgecolor='#FFFFFF', facecolor='#BEFFF0', linestyle='dashdot')
ax17.fill_between(np.log10(totala), np.log10(totala**3*early_pbh0), np.log10(totala**3*early_pbh1), label="PBHs (f=1)", facecolor="#DCF6FF")
ax17.plot(np.log10(totala), np.log10(totala**3*early_pbh2), label="Largest PBH (f=0.3)")
ax17.plot(np.log10(totala), np.log10(totala**3*early_pbh4), label=r"Evaporating PBH ($M=5\times 10^{-20} M_{\odot}$)")
ax17.plot(np.log10(totala), np.log10(totala**3*hawking4), label="Hawking Radiation", c='#FF00FF', linestyle='dashed')
ax17.axvline(x=np.log10(1/(1+zeq)), c='#000000', linestyle='dotted', label=r'Matter-Radiation Equality')
ax17.axvline(x=np.log10(1/(1+zrec)), c='#8B1EB3', linestyle='dotted', label=r'Recombination')
ax17.axvline(x=np.log10(1/(1+zdec)), c='#EE4266', linestyle='dotted', label=r'Decoupling')
ax17.axvline(x=np.log10(a[z30]), c='#FFD23F', linestyle='dotted', label=r'Cosmic Dawn')
ax17.axvline(x=np.log10(a[strpeak_arg]), c='#50E1BF', linestyle='dotted', label=r'SFR Peak')
ax17.axvline(x=np.log10(a[lambdadom_arg]), c='#0EAD69', linestyle='dotted', label=r'$\Lambda$ Domination')
ax17.set_xlabel('Scale Factor')
ax17.set_ylabel('Entropy Density')
ax17.set_title(r'Log-Log Entropy Density vs Scale Factor')
ax17.set_ylim(bottom=-25, top=25)
ax17.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
fig17.savefig("conformal_s_evolution_full.eps")


print("Present Day Entropy:  S[k] from this code (Egan & Lineweaver 2009)")
print("-------------------")
print("Cosmic Event Horizon       : %10.2E \t(2.6E+122)" % (s_ceh[-1]*volumeceh[-1]/kb))
print("Supermassive Black Holes   : %10.2E \t(1.2E+103)" % (s_smbh[-1]*volumeceh[-1]/kb))
print("Stellar Mass Black Holes   : %10.2E \t(2.2E+96)" % (s_bh[-1]*volumeceh[-1]/kb))
print("Cosmic Microwave Background: %10.2E \t(2.03E+88)" % (s_cmb[-1]*volumeceh[-1]/kb))
print("Primordial Neutrinos       : %10.2E \t(1.93E+88)" % (s_nu[-1]*volumeceh[-1]/kb))
print("Gas and Dust               : %10.2E \t(2.7E+80)" % (s_bm[-1]*volumeceh[-1]/kb))