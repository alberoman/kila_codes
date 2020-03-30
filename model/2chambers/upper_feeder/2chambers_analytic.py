#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:39:49 2020

@author: aroman
"""


import matplotlib.pyplot as plt
import numpy as np
from TwoChambers_solvers import  *


rho = 2600
g = 9.8
#Conduit parameters
condd = 5
conds = 4
ls = 3e+4
ld = 3.5e+3
mu = 100

#Chambers parameters
Vs = 5e+9
Vd  = 8e+9
ks = 1e+9
kd = 1e+9
#Pressures and friction6
pt = 15e+6
taus = 5e+6
taud = 3e+6

#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
#Calculating parameters

#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
#Calculating parameters

R1 = rho * g * Vs /(ks*S)
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
PS0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
PD0 = min((PS0* (1 + T1) + T1*R3),0) 
w0 = np.array([PS0,PD0])
ps = []
pd = []
t = []
TSLIP = 0
timescale = np.max(np.array([T1,1]))
TSLIP_seed = 1
N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
for i in range(1,N_cycles + 1):
    tseg,tslip,PS,PD,pd0 = system_2chambers(w0,params,PSLIP,TSLIP,TSLIP_seed)
    TSLIP_seed = tslip
    ps.append(PS)
    pd.append(PD)
    t.append(tseg + tcounter)
    PS0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
    PD0 = pd0
    PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
    w0 = np.array([PS0,PD0])
    tcounter = tcounter + tslip
ps = np.concatenate((ps)) * taus
pd = np.concatenate((pd)) * taus
t = np.concatenate((t)) * tstar
plt.figure(1)
plt.plot(t / (3600* 24),pd / 1e+6,'blue')
plt.plot(t / (3600* 24),ps / 1e+6,'red')
plt.ylabel('Pressure deep chamber [MPa]')
plt.xlabel('Time [Days]')
plt.legend(['Deep','Shallow'])
