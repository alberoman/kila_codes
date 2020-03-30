#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:13:43 
his code compare the performance in time and accuracy of the direct model for two chambers. With the first method the system of equations is integrated numerically with RK4, with the 
second method the analytical solution is used for each cycle. This second method uses scipy.optimize.newton to find the time at which the piston slips. @author: aroman
"""
import matplotlib.pyplot as plt
import numpy as np
from TwoChambers_solvers import  *

import time as tm
rho = 2600
g = 9.8
#Conduit parameters
condd =  2
conds = 3
ls = 3e+4
ld = 2.5e+3
mu = 100

#Chambers parameters
Vs = 2e+9
Vd  = 7e+9
ks = 1e+9
kd = 1e+9
#Pressures and friction
pt = 10e+6
taus = 3e+6
taud = 2e+6

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
n_cycle = 1
tcounter = 0

ps0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
pslip = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
pd0 = min((ps0* (1 + T1) + T1*R3),0) 
w0 = np.array([ps0,pd0]) 
ps_RK = []
pd_RK = []
time_RK =[]

#Numerical parameters RK
tol = 1e-4 #Error tolerance parameters on Newton-Raphson
tendRK = 1000 # Normalized time
dtRK = 1 * 1e-3
begin = tm.time()
while pslip > - R3:
    tstick,pShal,pDeep = system_2chambers_RK4(w0,tendRK,dtRK,params,pslip,tol)
    time_RK.append(tstick + tcounter)
    ps_RK.append(pShal)
    pd_RK.append(pDeep)
    
    tcounter = tcounter + tstick[-1]
    n_cycle = n_cycle + 1
    #Recalculating initial conditions, careful they are here dimensiona;l
    ps0 =  - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * (n_cycle - 1)
    pslip =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * n_cycle
    pd0 = pDeep[-1] 
    w0 = np.array([ps0,pd0]) 
    #Scale

ps_RK = np.concatenate((ps_RK)) * taus
pd_RK = np.concatenate((pd_RK)) * taus
time_RK = np.concatenate((time_RK)) * tstar
end = tm.time()
print('Time exectution numerical solution: ',end-begin,' seconds')
#Calculate using analytical solutions and Newton-Raphson from scipy to find the time at which piston slip
#Careful, in this case
A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
B = -T1/2 - PHI/2 - 1./2
C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
D = T1/2 - PHI /2 + 1./2

n_cycle = 1
tcounter = 0
params = [R3,T1,PHI,A,B,C,D]
PS0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
PD0 = min((PS0* (1 + T1) + T1*R3),0) 

w0 = np.array([PS0,PD0])
PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
PS = np.ones(len(time_RK))
PD = np.ones(len(time_RK))
time = time_RK /tstar
TSLIP = 0
TSLIP_seed = 1. / T1
begin = tm.time()

while PSLIP > - R3:   
     tslip,PS,PD,pd0 = system_2chambers_timein(w0,params,PSLIP,TSLIP,time,PS,PD,TSLIP_seed)
     TSLIP_seed = tslip
     TSLIP = TSLIP + tslip
     n_cycle = n_cycle + 1
     PS0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * (n_cycle - 1)
     PD0 = pd0
     PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * n_cycle
     w0 = np.array([PS0,PD0])
end = tm.time()
print('Time exectution analytical solution: ',end-begin,' seconds')
ps_analytical = PS * taus
pd_analytical = PD * taus
time = time * tstar
plt.figure(1)
plt.plot(time / (3600*24),pd_RK / 1e+6,'b')
plt.plot(time / (3600* 24),pd_analytical / 1e+6,'cyan')
plt.ylabel('Pressure deep chamber [MPa]')
plt.xlabel('Time [Days]')
plt.legend(['RK','Analytic'])
plt.figure(2)
plt.plot(time / (3600*24),ps_RK / 1e+6,'r')
plt.plot(time / (3600* 24),ps_analytical / 1e+6,'orange')
plt.ylabel('Pressure shallow chamber [MPa]')
plt.xlabel('Time [Days]')
plt.legend(['RK','Analytic'])
print('DONT WORRY if analytic solution is offset and has an irregular shape. It is calculated on the time basis of the numerical which has a limited accuracy for predicting the time at which slip start. The analytic solution is still more accurate')

