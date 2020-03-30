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

#Initial pressure in the lower chamber
pd0_initial = -2

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
PS0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
PD0 = pd0_initial
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
#plt.figure(1)
plt.plot(t / (3600* 24),pd / 1e+6,'blue')
plt.plot(t / (3600* 24),ps / 1e+6,'red')
plt.ylabel('Pressure chamber [MPa]')
plt.xlabel('Time [Days]')
plt.legend(['With Piston collapse,feeding eruption','Without piston collapse, nofeeding'])

#Calculating approximation
tcounter = 0
params = [R1,R3,R5,T1,PHI]
PS0 =  - 1  + 2 * (1 - R5) / (1 + R1) 
PD0 = pd0_initial
w0 = np.array([PS0,PD0])
ps = []
pd = []
t = []
N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) 
for i in range(1,N_cycles):
    tseg,tslip,PS,PD,pd0 = system_2chambers_smallVar(w0,params,i)
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
plt.plot(t / (3600* 24),pd / 1e+6,'orange')
plt.plot(t / (3600* 24),ps / 1e+6,'cyan')
    