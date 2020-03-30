#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:00:55 2019

@author: aroman
"""

import numpy as np
import matplotlib.pyplot as plt
from solvers_piston import *

exp  = np.linspace(-4,0)
R1 = 0.02
R2 = 1e-6
R3 = 1.1
R5 = 0.7
R4 = -1
x0 = 0

DeltaP,x,velocity,N_cycle,time,t_slip,period_stick_slip,dx_slip,slip_duration,q_out,vol_out = isostatic_piston(x0,R1,R2,R3,R4,R5)

sticks = period_stick_slip - slip_duration 
n = np.linspace(1,len(period_stick_slip),len(period_stick_slip))
nminus = n - 1 
sticks_th =- 1. /R2 * np.log((R3 * (1+R1) - 2*n*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*nminus) - (1+R1)))   
plt.figure(1)
plt.plot(n,sticks,'o')
plt.plot(n,sticks_th,'o')
plt.show()
