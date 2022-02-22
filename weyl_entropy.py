# -*- coding: utf-8 -*-
"""
Weyl curvature entropy calculator

General plan:
-Use CAMB to find perturbation metric from early universe (scalar, vector, tensor perturbations?)
-From the primordial perturbations metric, calculate Weyl tensor and Riemman tensor (can we get this from Weyl potential?)
-Determine how to evolve Weyl tensor forward in time from primordial power spectrum
-Calculate the Weyl entropy from possible formulas from Weyl entropy paper (Weyl scalar over Riemann scalar?)

Now find entropy of graviton gas
-Take the primordial power spectrum again, convert gravitational wave background into graviton gas (wavenumber, see Egan & Lineweaver)
-Calculate entropy with Sakur-Tetrode formula

QUESTIONS:
    -how do density perturbations relate to metric perturbations? sigma is d(rho)/rho, can this be related to stress-energy?
"""

# import sys, platform, os
# print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
# sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from matplotlib import pyplot as plt

from camb.symbolic import *

from camb import model, initialpower, get_matter_power_interpolator

sympy.init_printing()
print('CAMB: %s, Sympy: %s'%(camb.__version__,sympy.__version__))

params=camb.read_ini('CAMB/inifiles/planck_2018.ini')

PK=get_matter_power_interpolator(params)
print('Power spectrum at z=0.5, k/h=0.1/Mpc is %s (Mpc/h)^3 '%(PK.P(0.5, 0.1)))

#plt.plot(PK)