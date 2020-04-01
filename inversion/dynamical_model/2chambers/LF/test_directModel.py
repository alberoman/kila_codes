#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:38:15 2020

@author: aroman
"""

import numpy as np
import matplotlib.pyplot as plt
from main_lib import DirectModelEmcee_inv_LF_test
tend = 3600 * 24 * 200
tTilt = np.linspace(0,tend,1000)
tGPS = tTilt[::25]
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 4e+4
ld = 2.5e+3
mu = 100
#Pressures and friction
pt = 2.0e+7
poisson = 0.25
lame = 1e+9
Rcyl = 1.0e+3
S = 3.14 * Rcyl**2 
const = -9. /(4 * 3.14) * (1 - poisson) / lame
VsExpSamp = 9
VdExpSamp = 10
kExpSamp = 9
pspdSamp = 5e+6
R3Samp = 5
condsSamp = 4
conddSamp = 4
ps,pd,GPS = DirectModelEmcee_inv_LF_test(tTilt,tGPS,
                                              VsExpSamp,VdExpSamp,kExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                                              ls,ld,pt,mu,
                                              rhog,const,S)
