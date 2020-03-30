#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:18:38 2020

@author: aroman
"""

import numpy as np
import theano
import theano.tensor as T



# define a named function, rather than using lambda



Nmax = 30
rho = 2600
g = 9.8
#Conduit parameters
condd = 2
conds = 3
ls = 3e+4
ld = 2.5e+3
mu = 100

#Chambers parameters
Vs = 2e+9
Vd  = 4e+10
ks = 1e+9
kd = 1e+9
#Pressures and friction
pd0 = 0e+6 # Pressure in the deep chamber
pt = 15e+6
taus = 3e+6
taud = 1e+6
pspd = taus - taud
#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
#Calculating parameters
R1Mod = rho * g * Vs /(ks*S)
T1Mod = (conds / condd )**4 * ld /ls
phiMod = kd /ks * Vs / Vd
low_bound = 2 * (Nmax + 2) * R1Mod / (1 + R1Mod) * pspd
plps = low_bound
n = np.arange(1,Nmax+1)
t = np.linspace(0,1e+7,600)




 