from matplotlib import pyplot as plt
import sys, platform, os
import matplotlib
import numpy as np

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#Now get matter power spectra and sigma8 at redshift 0 and 0.8
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[100], kmax=1e4)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e4, npoints = 200)

#Non-Linear spectra (Halofit)
#results.calc_power_spectra(pars)

for i, (redshift, line) in enumerate(zip(z,['b','r', 'g','c','y'])):
    plt.loglog(kh, pk[i,:], color=line)
plt.xlabel('k/h Mpc');
plt.legend(['z=0','z=1','z=10','z=100','z=1000'], loc='lower left');
plt.title('Matter power at z=%s and z= %s'%tuple(z));

def tophat(k, M):
    return 3*(np.sin(k*M)-k*M*np.cos(k*M))/(k*M)**3
def b_z(z):
    return 1
sigsq=[] #Function of z and M