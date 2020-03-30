#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:17:50 2019

@author: aroman
"""
import numpy as np
from solvers_piston import *
from python_version import *
import time as chrono

def calc_cycles(R1,R2,R3,R5,N_cycles):
    x = np.array([0])
    time = np.array([0])
    press = np.array([-1])
    volume = np.array([0])
    omega = (1 + R1)**0.5
    N_cycles_predicted = np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))
    N_loops = int(np.min([N_cycles,N_cycles_predicted]))
    N_points_slip = 100
    N_points_stick = 100
    t_slip  = np.linspace(3.14 /np.sqrt(1 + R1)/N_points_slip, 3.14 /np.sqrt(1 + R1),N_points_slip)
    for i in range(1,N_loops + 1):
        #slip phase
        x_slip = x[-1] + (R5 - 1) / (1 + R1) * np.cos(omega * t_slip) + (1 - R5)/(1 + R1) 
        p = - 1 - R1 * x[-1] + x_slip - x[- 1] 
        x = np.concatenate((x,x_slip),axis = 0)
        press = np.concatenate((press,p),axis = 0)
        time = np.concatenate((time,time [-1] + t_slip),axis = 0)
        if i == N_loops:
            t_end_stick = 2 * (- 1. /R2 * np.log((R3 * (1+R1) - 2*(i-1)*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*(i - 2)) - (1+R1))))
        else:
            t_end_stick =  - 1. /R2 * np.log((R3 * (1+R1) - 2*i*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*(i - 1)) - (1+R1)))  
        t_stick = np.linspace(t_end_stick/N_points_stick,t_end_stick,N_points_stick)
        p = (press[-1] + R3) * np.exp(-R2 * t_stick) - R3
        x_stick= x[-1] * np.ones(len(p))
        press = np.concatenate((press,p),axis = 0)
        x=np.concatenate((x,x_stick),axis = 0)
        time = np.concatenate((time,time [-1] + t_stick), axis = 0)
    return x,press,time

def stick_slip(n,pressure,tslip,dtslip,time,R1,R2,R3,R5,freq):
    if n < 2:
        tslip = 0
    else:
        tslip = tslip + dtslip - 1. /R2 * np.log((R3 * (1+R1) - 2*(n - 1)*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*(n - 2)) - (1+R1)))
    tstick  = tslip + dtslip
    pressure[time > tslip] = 2 * (n -1) * (1 - R5) + (R5 -1)/(1 + R1) * np.cos(freq * (time[time > tslip] -tslip)) + (1 - R5)/(1 + R1) #slip phase pressure
    pressure[time > tstick] = (-R1 * 2 * (n - 1) * (1 - R5) / (1 + R1) -1 + R3) * np.exp(-R2 * (time[time > tstick]-tstick)) 
    return pressure,tslip

R1 = 0.05
R2 = 1e-7
R3 = 2.0
R5 = 0.7
R4 = -1
x0 = 0
cycles = 1000
#Numerical solution
t1 = chrono.time()
x_analytic,DeltaP_analytic,t_analytic = calc_cycles(R1,R2,R3,R5,cycles)
t2 = chrono.time()
dt_analytic = t2 - t1
#Numeric solution
t1 = chrono.time()
DeltaP_numeric,x_numeric,velocity,N_cycle,t_numeric,t_slip,period_stick_slip,dx_slip,slip_duration,q_out,vol_out,p_slip,dp_slip = isostatic_piston(x0,R1,R2,R3,R4,R5)
t2 = chrono.time()
dt_numeric = t2 - t1
#Optimized theano oriented solution
t1 = chrono.time()
DeltaP_theano_or = direct_model(t_analytic,R1,R2,R3,R5)
t2 = chrono.time()
dt_theano_or = t2 - t1
print('Time anaytic sol = ' , dt_analytic)
print('Time numerical sol = ' , dt_numeric)
print('Time theano_oriented sol = ' ,dt_theano_or)

print('Ratio analytic/numeric = ',dt_analytic/dt_numeric )
plt.plot(t_analytic,DeltaP_analytic,'r')
plt.plot(t_numeric,DeltaP_numeric,'b')
plt.plot(t_analytic,DeltaP_theano_or,'g')
plt.show()
