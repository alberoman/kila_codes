#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:02:28 2019

@author: aroman
"""

import numpy as np  
import matplotlib.pyplot as plt
from solvers_piston import *
'''
Parameters
'''
x0 = 0
g = 9.8
rho = 2700
#chamber and elasticity
V0= 1e+9
k = 1.2e+9
#Piston geometrical parameters
Rp = 7e+2
hp = 5e+2
S = 3.14 * Rp**2
Vp = S * hp
m = rho * Vp
#Conduit and mama properties
a = 4.
mu = 5e+2
l = 3e+4
#Friction properties
c0 = 10e+6
fric_coef = 0.01
Fs = c0 * S
R5 = 0.1
Fd = Fs / R5
#Driving pressure
Pl =2.0e+7
tstar = ((m*V0) / (k* S**2))**0.5
xstar = Fs*V0/(k*S**2)
qstar = Fs *(V0/(k*m))**0.5
Vstar = xstar * S
R1 = rho * g * V0 /(k*S)
R2 = (m * k / V0)**0.5 *(3.14 * a**4 )/(8* mu * l* S) 
R3 = Pl / Fs * S
R4 = -1
x0 = x0/xstar
#R1 = 0.6
DeltaP,x,velocity,N_cycle,time,t_slip,dt_slip,dx_slip,slip_duration,q_out,vol_out,p_slip,dp_slip = isostatic_piston(x0,R1,R2,R3,R4,R5)


'''
Upscale
'''
time = time * tstar
t_slip = t_slip * tstar
dt_slip = dt_slip * tstar 
DeltaP = DeltaP * Fs/S
dp_slip = dp_slip * Fs/S
p_slip = p_slip * Fs/S
x = x * xstar
dx_slip = dx_slip *xstar
q = 3.14 * a**4 / (8 * mu * l) *(DeltaP+Pl)
dt_cycle = np.diff(t_slip)
print('Number of stick-slio events: ', N_cycle)
print('Ratio of duration between last and first cycle: ',dt_cycle[-1]/dt_cycle[0])
dt = np.diff(time)
Vp = x*S
q_out = q_out * qstar
vol_out = vol_out *Vstar
V = np.zeros(len(time))
Vc = 0

for i in range(len(dt)):
    dV = q[i]*dt[i]
    Vc = Vc + dV
    V[i+1]= Vc 
tmin = np.min(time)
tmax = np.max(time) 
n = np.arange(1,len(dx_slip) +1) 
dx_slip_analytic = 2 * R1/(rho*g)*(c0 - R5*c0) / (1 + R1) * np.ones(len(n))
dp_slip_analytic = 2*(c0 - R5*c0) / (1 + R1) * np.ones(len(n))
p_slip_analytic =  - c0 -  2 * R1*(c0 - R5*c0) / (1 + R1) * (n - 1)
tau = 8 * mu *l * V0 / (3.14 * a**4 * k)
phi = (Pl - c0) /(c0 - R5 * c0)
dt_analytic = -tau * np.log(((1 + R1)*phi - 2 * n * R1 ) / (((1 + R1)*phi - 2 *(R1*(n -1) -1))))
n_analytic = (1 + R1) * phi / (2*R1)
print(n_analytic,len(n))
fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize = (9,10))
ax[0].plot(n,dx_slip,'ro')
ax[0].plot(n,dx_slip_analytic,'bo')
ax[1].plot(n,dp_slip,'ro')
ax[1].plot(n,dp_slip_analytic,'bo')
ax[2].plot(n,p_slip,'ro')
ax[2].plot(n,p_slip_analytic,'bo')
ax[3].plot(n[0:-1],dt_slip,'ro')
ax[3].plot(n,dt_analytic,'bo')
plt.show()