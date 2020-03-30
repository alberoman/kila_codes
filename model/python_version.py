#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:40:41 2019

@author: alb
"""


import numpy as np

def direct_model(t,R1,R2,R3,R5):
    pressure = np.ones(len(t)) #init pressure array
    N = np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))
    loops = np.arange(1,N+1)
    freq = np.sqrt(1 + R1)
    dtslip = 3.14 / freq #deltaT slip
    tslip  = 0. #first time  slip
    frac = (1 - R5)/(1 + R1)
    one_plus_r1 = 1 + R1
    one_min_r5 = (1 - R5)
    for i in loops:
        if i < 2:
            tslip = 0
        else:
            tslip = tslip + dtslip - 1. /R2 * np.log((R3 * one_plus_r1 - 2*(i - 1)*R1*one_min_r5 - one_plus_r1) / (R3 * one_plus_r1 - 2 *one_min_r5*(-1+R1*(i - 2)) - one_plus_r1))
        tstick  = tslip + dtslip
        pressure[t >= tslip] = frac * (1 - 2*R1*(i - 1) - np.cos(freq * (t[t >= tslip] - tslip))) -1
        pstick  = 2 * frac * (1- R1*(i - 1)) -1
        pressure[t > tstick] = (pstick + R3) * np.exp(-R2 * (t[t > tstick]-tstick)) - R3
    return pressure