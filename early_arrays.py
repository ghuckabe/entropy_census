#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:03:54 2024

@author: gabbyhuckabee
"""

import numpy as np
import constants as c

#Format of these arrays is [[eps1, temp1, s1], [eps2, tem2, s2], ...]
early_pretop=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prebottom=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_precharm=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prestrange=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_preup=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_predown=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_pregluon=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_pretau=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_premuon=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_preelectron=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_preneutrino=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prew=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prez=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prephoton=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_prehiggs=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]

early_posttop=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postbottom=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postcharm=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_poststrange=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postup=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postdown=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postgluon=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_posttau=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postmuon=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postelectron=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postneutrino=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postw=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postz=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postphoton=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_posthiggs=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postchargedpion=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]
early_postneutralpion=[np.zeros(c.early_length), np.zeros(c.early_length), np.zeros(c.early_length)]

early_baryon=np.zeros(c.early_length)
early_dm=np.zeros(c.early_length)
early_l=np.zeros(c.early_length)
early_proton=np.zeros(c.early_length)
early_neutron=np.zeros(c.early_length)
early_hydrogen=np.zeros(c.early_length)

protontemp=np.zeros(c.early_length)
neutrontemp=np.zeros(c.early_length)

early_mup=np.zeros(c.early_length)
early_mun=np.zeros(c.early_length)

early_s_proton=np.zeros(c.early_length)
early_s_neutron=np.zeros(c.early_length)
early_s_hydrogen=np.zeros(c.early_length)
early_s_baryon=np.zeros(c.early_length)
early_s_dm=np.zeros(c.early_length)
early_s_l=np.zeros(c.early_length)

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