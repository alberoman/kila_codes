#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:23:23 2020

@author: aroman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:13:43 
Make a movie of a two source model with piston collapse
@author: aroman
"""
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar # For dealing with Colorbars the proper way - TBD in a separate PyCoffee ?
import time as tempo
import numpy as np

condd = 3
conds = 4
ls = 3e+4
ld = 2.5e+3
Vs = 2e+9
Vd  = 8e+9
ks = 1e+9
kd = 1e+9
pd0 = 1e+6
pd0_lin = 0e+7
pt = 8.0e+6
mu = 100


taus = 4e+6
taud = 1e+6
rho = 2600
g = 9.8
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
R1 = rho * g * Vs /(ks*S)
T1 = (conds / condd )**4 * ld /ls
un_plus_T1 = 1 + T1
un_plus_R1 = 1 + R1

phi = kd /ks * Vs / Vd
tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
ps_analytic = []
pd_analytic = []
time =[]
n = 1
numerator = taus + R1 * n * (taus - taud) / un_plus_R1 + pd0 / un_plus_T1 -pt * T1 / un_plus_T1
denominator = taus + R1 * (n-1) * (taus - taud) / un_plus_R1 - 2 * (taus - taud) / un_plus_R1 + pd0 / un_plus_T1 -pt * T1 / un_plus_T1
tend = - tstar / un_plus_T1 * np.log(numerator/denominator)
tend_lin = tend
t0 = 0
t0_lin = 0
time0 = tempo.time()
while not np.isnan(tend):
#while n < 3:
    print(n)
    t = np.linspace(0,tend,15)
    ps0 = - taus - (n -1) * R1 * (taus - taud) / un_plus_R1 + 2 * (taus - taud) / (1 + R1)
    ps = pd0 * 1 / un_plus_T1 - pt * T1 / un_plus_T1 + (ps0 - pd0 * 1 / un_plus_T1 + pt * T1 / un_plus_T1) * np.exp( - un_plus_T1 * t/tstar )
    pd = pd0 + kd * 3.14 * condd**4 / (Vd * 8 * mu * ld)*(-pd0 + pd0 / un_plus_T1 - pt * T1 / un_plus_T1) * t  
    + phi / un_plus_T1 * (ps0 - pd0 / un_plus_T1 + pt * T1 / un_plus_T1) * (np.exp(- un_plus_T1 * t /tstar) - 1)
    t = t + t0
    pd0 = pd[-1]
    t0 = t[-1] 
    numerator = taus + R1 * n * (taus - taud) / un_plus_R1 + pd0 / un_plus_T1 -pt * T1 / un_plus_T1
    denominator = taus + R1 * (n-1) * (taus - taud) / un_plus_R1 - 2 * (taus - taud) / un_plus_R1 + pd0 / un_plus_T1 -pt * T1 / un_plus_T1
    tend = - tstar / un_plus_T1 * np.log(numerator/denominator)
    #print(numerator),print(denominator)
    ps_analytic.append(ps)
    pd_analytic.append(pd)
    time.append(t)
    n = n + 1
time1 = tempo.time()
#print(time1 - time0)
time = np.concatenate(time,axis = 0)

ps = np.concatenate(ps_analytic,axis = 0)
ps = np.reshape(ps,(len(ps),1))
pd = np.concatenate(pd_analytic,axis = 0)
pd = np.reshape(pd,(len(pd),1))
pressure = np.concatenate((ps - ps[0],pd - pd [0]),axis = 1)
V = np.array([Vs,Vd])
poisson = 0.25
lame = 1e+9


xstation1 = 80
ystation1 = 120
xstation2 = 80
ystation2 = 80
