#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:26:34 2019

@author: aroman
t"""
import numpy as np
import theano.tensor as tt
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

