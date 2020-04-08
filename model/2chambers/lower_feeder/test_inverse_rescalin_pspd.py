#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:39:49 2020

@author: aroman
"""


import matplotlib.pyplot as plt
import numpy as np
from TwoChambers_solvers import  *
from main_lib import DirectModelEmcee_inv_debug

rho = 2600
g = 9.8
#Conduit parameters
condd = 4.
conds = 4
ls = 3e+4
ld = 2.5e+3
mu = 200

#Chambers parameters
Vs = 3e+10
Vd  = 1e+9
ks = 1e+9
kd = 1e+9
#Pressures and friction6
pt = 15e+6
taus = 5e+6
taud = 1e+6

#Cylinder parameters
Rcyl = 4.0e+2
S = 3.14 * Rcyl**2 
#Calculating parameters


#Calculating parameters

R1 = rho * g * Vd /(kd*S)
T1 = (conds / condd )**4 * ld /ls
PHI = kd /ks * Vs / Vd
R5 = taud / taus
R3 = pt  / taus
params = [T1,PHI,R3] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
un_plus_T1 = 1 + T1
un_plus_R1 = 1 + R1

#Initialization
tcounter = 0

A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
B = -T1/2 - PHI/2 - 1./2
C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
D = T1/2 - PHI /2 + 1./2

params = [R3,T1,PHI,A,B,C,D]
TSLIP = 0
TSLIP_seed = 1
tseg,tslip,PS,PD,ps0 = system_2chambers(np.array([0,0]),params,-1,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
PS0 =  ps0
PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
w0 = np.array([PS0,PD0])
ps = []
pd = []
t = []
timescale = np.max(np.array([T1,1]))
N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
for i in range(1,N_cycles + 1):
    tseg,tslip,PS,PD,ps0 = system_2chambers(w0,params,PSLIP,TSLIP,TSLIP_seed)
    TSLIP_seed = tslip
    ps.append(PS)
    pd.append(PD)
    t.append(tseg + tcounter)
    PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
    PS0 = ps0
    PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
    w0 = np.array([PS0,PD0])
    tcounter = tcounter + tslip
ps = np.concatenate((ps)) * taus
pd = np.concatenate((pd)) * taus
t = np.concatenate((t)) * tstar

VsExpSamp = np.log10(Vs)
VdExpSamp = np.log10(Vd)
kExpSamp = np.log10(ks)
pspdSamp = taus - taud
R3Samp = (pt - taus) / (pspdSamp)
condsSamp = conds
conddSamp = condd
rhog = rho * g
cs = 1
psInv,pdInv = DirectModelEmcee_inv_debug(t,
                         VsExpSamp,VdExpSamp,kExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                         ls,ld,pt,mu,
                         rhog,cs,S)







plt.figure(1)
plt.plot(t / (3600* 24),(pd + taus)/ 1e+6,'bo')
plt.plot(t / (3600* 24),(pdInv)/ 1e+6,'co')
#plt.plot(t / (3600* 24),(ps + taus)/ 1e+6,'red')
#plt.plot(t / (3600* 24),(psInv)/ 1e+6,'orange')

plt.ylabel('Pressure deep chamber [MPa]')
plt.xlabel('Time [Days]')
plt.legend(['With Piston collapse, not feeding eruption','Without piston collapse,feeding'])
