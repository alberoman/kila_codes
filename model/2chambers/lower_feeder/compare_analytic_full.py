#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:39:49 2020

@author: aroman
"""


import matplotlib.pyplot as plt
import numpy as np
from TwoChambers_solvers import  *
import time

rho = 2600
g = 9.8
#Conduit parameters
condd = 8
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
pt = 10e+6
taus = 2.5e+6
taud = 1e+6

#Cylinder parameters
Rcyl = 4.0e+2
S = 3.14 * Rcyl**2 
m = 2600 * S * 1000
#Calculating parameters


#Calculating parameters

R1 = rho * g * Vd /(kd*S)
T1 = (conds / condd )**4 * ld /ls
PHI = kd /ks * Vs / Vd
R5 = 0
R3t = (pt - taus)  / (taus -  taud)
R3  = 3.14  * condd**4 / ( 8 * mu * ld *S ) * (ks * m / Vs)**0.5
R4  = 3.14 * condd**4 / (mu * ld *S ) * (Vs/Vd**2 * kd**2/ks * m)**0.5
params = [T1,PHI,R3t] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
xstar = (taus - taud) * Vd / (kd * S) 
un_plus_T1 = 1 + T1
un_plus_R1 = 1 + R1
m = 2600 * S * 1000
#Initialization
tcounter = 0

A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
B = -T1/2 - PHI/2 - 1./2
C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
D = T1/2 - PHI /2 + 1./2

params = [R3t,T1,PHI,A,B,C,D]
PD0 =    + 2 * (1 - R5) / un_plus_R1 
PSLIP =  - 2 * R1 * (1 - R5)/un_plus_R1
PS0 = -1.5
w0 = np.array([PS0,PD0])
ps = []
pd = []
t = []
TSLIP = 0
timescale = np.max(np.array([T1,1]))
TSLIP_seed = 1
N_cycles = int((1 + R1)/ (2 * R1 ) * R3t) - 1
time1 = time.time()
for i in range(1,N_cycles + 1):
    
    tseg,tslip,PS,PD,ps0 = system_2chambers(w0,params,PSLIP,TSLIP,TSLIP_seed)
    TSLIP_seed = tslip
    ps.append(PS)
    pd.append(PD)
    t.append(tseg + tcounter)
    PD0 =   + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
    PS0 = ps0
    PSLIP =  - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
    w0 = np.array([PS0,PD0])
    tcounter = tcounter + tslip
    print(i)
psan = np.concatenate((ps)) * (taus - taud)
pdan = np.concatenate((pd)) * (taus - taud)
tan = np.concatenate((t)) * tstar
time2 = time.time()
dtan =time2 - time1

#plt.figure(1)
#plt.plot(tan / (3600* 24),pdan / 1e+6,'blue')
#plt.plot(tan / (3600* 24),psan / 1e+6,'red')
#plt.ylabel('Pressure deep chamber [MPa]')
#plt.xlabel('Time [Days]')
#plt.legend(['With Piston collapse, not feeding eruption','Without piston collapse,feeding'])
ps = []
pd = []
x = []
t = [] 
x0 = 0
pd0 = 0
ps0 = -1.5
dx,vfinal,PD0,PS0 = slip_phase(R1,R3,R4,0,pd0,ps0)
xcumul = dx
w0 = np.array([PS0,PD0])
PSLIP = -R1 * xcumul
tcounter = 0
print('new')
i = 1
time1 = time.time()

while (np.abs(PSLIP) < R3t):
    w0 = np.array([PS0,PD0])
    tseg,tslip,PS,PD,ps0 = system_2chambers(w0,params,PSLIP,TSLIP,TSLIP_seed)
    x.append(xcumul * np.ones(len(PS)))
    ps.append(PS)
    pd.append(PD)
    t.append(tseg + tcounter)
    tcounter = tcounter + tslip
    pd0 = PSLIP
    ps0 = PS[-1]
    dx,vfinal,PD0,PS0 = slip_phase(R1,R3,R4,xcumul,pd0,ps0)
    xcumul = xcumul + dx
    PSLIP = -R1 * xcumul
    print(i)
    i = i + 1
psnum = np.concatenate((ps)) * (taus - taud)
pdnum = np.concatenate((pd)) * (taus - taud)
tnum = np.concatenate((t)) * tstar
xnum = np.concatenate((x)) * xstar

time2 = time.time()
dtnum = time2 -  time1
plt.plot(tnum / (3600* 24),pdnum / 1e+6,'c')
plt.plot(tnum/ (3600* 24),psnum / 1e+6,'orange')
print('Time execution analytic is ', dtan)
print('Time execution numerical is ', dtnum)













    
    
        


    