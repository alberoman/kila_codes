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
from inversion_lib import *
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
g = 9.8
rho = 2700
#chamber and elasticity
V0= 1e+9
radius_cube = 0.75 * V0 / 3.14
#Elastic properties crust
poisson = 0.25
lame = 1e+9
#Position of the chamber
radial_distance = 1000
depth  = 2000
R = (radial_distance**2 + depth**2)**0.5 
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
Fs = c0 * S
R50 = 0.1
Fd = Fs / R50
#Driving pressure
Pl =2.0e+7
tstar = ((m*V0) / (k* S**2))**0.5
xstar = Fs*V0/(k*S**2)
pstar = Fs/S

qstar = Fs *(V0/(k*m))**0.5
Vstar = xstar * S
R10 = rho * g * V0 /(k*S)
R20 = (m * k / V0)**0.5 *(3.14 * a**4 )/(8* mu * l* S) 
R30 = Pl / Fs * S

#Creating synthetic data
dummy,t_synth = calc_cycles(R10,R20,R30,R50,cycles)
DeltaP_synth = piston_scaled(np.array([R10,R20,R30,R50,tstar]),t_synth)
DeltaP_synth = DeltaP_synth * Fs/S
t_synth = t_synth * tstar
#Adding some noise
