#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:02:30 2020

@author: aroman
"""

import numpy as np
from main_lib import *
from preparedata import *

rho = 2600
g = 9.8
#Conduit parameters
condd = 1.
conds = 3.5
ls = 3e+4
ld = 2.5e+3
mu = 200

#Chambers parameters
Vs = 2e+9
Vd  = 0.5e+10
ks = 1e+9
kd = 1e+9
#Pressures and friction6
pt = 25e+6
taus = 2.5e+6
taud = 1.5e+6

#Cylinder parameters
Rcyl = 4.0e+2
S = 3.14 * Rcyl**2 
pd0_initial = -2

VsExpSamp = np.log10(Vs)
VdExpSamp = np.log10(Vd)
pspdSamp = taus - taud
R3Samp = (pt - taus) / pspdSamp
deltap0Samp = pd0_initial + taus

ksExpSamp = np.log10(ks)
kdExpSamp = np.log10(kd)

rhog = rho * g
ps,pd = DirectModelEmcee_inv_UF_test(tTilt,deltap0Samp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,rhog,ls,ld,mu,S)