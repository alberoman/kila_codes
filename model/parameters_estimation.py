#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:11:28 2019

@author: aroman
"""

import numpy as np
import matplotlib.pyplot as plt

x0 = 0
g = 9.8
rho = 2700
#chamber and elasticity
#Piston geometrical parameters
Rp = 7e+2
hp = 7+2
S = 3.14 * Rp**2
Vp = S * hp
m = rho * Vp
#Conduit and mama properties
a = 4.
mu = 1e+2
l = 3e+4
#Friction properties
c0 =5.0e+6
Fs = c0 * S
v0 = 5e+9
R5 = 0.3
R3 = 1.3
k = np.linspace(1e+8,1e+10)
R1 = rho * g * v0 /(k*S)
R2 = (m * k / v0)**0.5 *(3.14 * a**4 )/(8* mu * l* S) 
tstar = ((m*v0) / (k* S**2))**0.5
xstar = Fs*v0/(k*S**2)
xslip = 1 * (1- R5)/ (1 + R1) * xstar
tslip = 3.14 / np.sqrt(1+R1) * tstar
dt_stick = -1./R2 * np.log((R3 * (1+R1) -2*R1*(1-R5) -(1+R1))/(R3 * (1+R1) +2*(1-R5) -(1+R1))) * tstar

plt.figure(1)
plt.semilogx(k,tslip)
plt.xlabel('k')
plt.ylabel('tslip')
plt.figure(2)
plt.semilogx(k,xslip)
plt.xlabel('k')
plt.ylabel('xslip')
plt.figure(3)
plt.semilogx(k,dt_stick/(3600*24))
plt.xlabel('k')
plt.ylabel('dtslip')
plt.show()
