#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:17:50 2019

@author: aroman
"""
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
from inversion_lib_scaled import *
import matplotlib.pyplot as plt

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
    return press,time




cycles = 1e+4

R10 = 0.05
R20 = 1e-6 
R30 = 1.8
R50 = 0.1

#Creating synthetic data
dummy,t_synth = calc_cycles(R10,R20,R30,R50,cycles)
DeltaP0 = piston_scaled(R50,t_synth)
#Adding some noise
DeltaP_Amp = 2 * (1 - R50) / ((1 + R10)) 
sigma_pert = DeltaP_Amp / 10
np.random.seed(42)
DeltaP_Pert = DeltaP0 + sigma_pert * np.random.randn(len(DeltaP0))
#Setup inversion

DeltaP = direct_model(piston_scaled,t_synth)

with pm.Model() as model:
#    R1 = pm.Uniform("R1", lower=0.01, upper=0.3)
#    R2 = pm.Uniform('R2',lower=1e-8, upper=1e-5)
#    R3 = pm.Uniform('R3',lower=1.1, upper=5.0)
    R5 = pm.Uniform('R5',lower=0.001, upper=1.0)
    Y_obs = pm.Normal('Y_obs', mu=DeltaP(R5), sd =  0.1, observed=DeltaP_Pert) 
    trace = pm.sample(5000)