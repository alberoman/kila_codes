#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:40:11 2019

@author: aroman
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def slip_phase(w,time,p):
    R1,R2,R3,R5 = p
    deriv = [ w[1],
    -R1*w[0] - w[2] - R5,
    w[1] - R2*w[2] - R3*R2]
    return deriv
def stick_phase(p_stick,t_stick,x_stick,dt,R1,R2,R3,R4,R5):
    if t_stick == 0 :
        t_slip = -1/R2 * np.log((R3 - 1) / (R3 + R4))
        time = np.np.linspace(t_stick,t_slip,1000)
        Delta_P = (R3+ R4) * np.exp(-R2*time) - R3
        flag = 0
    elif (R3 - R1 * x_stick - 1) < 0:
        print('Ending! No more underpressure available to drag the piston down',R3 - R1 * x_stick - R5)  
        time = np.linspace(t_stick,t_stick + 1./R2*1e+1,1000)
        Delta_P = (p_stick + R3) * np.exp(R2*t_stick)*np.exp(-R2*time) -R3 
        flag = 1
    else:
        t_slip = -1 / R2*np.log((R3 - R1 * x_stick - 1)/((p_stick + R3)*np.exp(R2*t_stick)))
        time = np.linspace(t_stick,t_slip,1000)
        Delta_P = (p_stick + R3) * np.exp(R2*t_stick)*np.exp(-R2*time) -R3
        flag = 0
    return time, Delta_P,flag

def isostatic_piston(x0,R1,R2,R3,R4,R5):
    ##### Here we calculate the first stick phase, ts is the moment at which the piston start to slip, at which Delta_P =  - R5
#R5 Fs/Fd for stick-slip you should have Fs/Fd> 1
#R3 PlS/Fd Contribution of the lithostatic to the outflow
# Having a time for stick implies R3 > R5 > 1

    t_start = 0
    p_start = -1
    x_start = x0
    v_start = 0
    dt = 0.01
    p = [R1,R2,R3,R5]
    t_slip_start = []
    slip_duration = []
    slip_duration = []
    pslip = []
    dpslip = []
    dx_slip = []
    tend =  5
    t_slip_phase = np.linspace(0,tend,1000)
    t_slip_start.append(0)
    w0 = [x_start,v_start,p_start]
    wsol = odeint(slip_phase, w0, t_slip_phase, args=(p,))
    velocity = wsol[:,1]
    v0 =velocity[velocity<0][0]
    err_vel = np.abs(v0)
    tend = t_slip_phase[velocity<0][0]
    t_slip_phase = np.linspace(0,tend,1000)
    wsol = odeint(slip_phase, w0, t_slip_phase, args=(p,))
    x = wsol[:,0]
    dx_slip.append(wsol[-1,0]-wsol[0,0])
    x_stick = x[-1]
    v = wsol[:,1]
    press = wsol[:,2]
    pslip.append(wsol[0,2])
    dpslip.append(wsol[-1,2] - wsol[0,2])
    
    t = t_slip_phase
    
    t_stick_phase, pres_stick,err_flag = stick_phase(press[-1],t[-1],x[-1],dt,R1,R2,R3,R4,R5)
    press = np.concatenate((press,pres_stick))
    x = np.concatenate((x,x_stick * np.ones(len(t_stick_phase))))
    v = np.concatenate((v,np.zeros(len(t_stick_phase))))

    t  = np.concatenate((t,t_stick_phase))
    N_cycle = 1
    while err_flag == 0 and N_cycle < 1000 :
        tend = t_stick_phase[-1] + 5
        t_slip_phase = np.linspace(t_stick_phase[-1],tend,1000)
        t_slip_start.append(t_slip_phase[0])
        w0 = [x[-1],v[-1],press[-1]]
        try:
            wsol = odeint(slip_phase, w0, t_slip_phase, args=(p,))
            velocity = wsol[:,1]
            v0 =velocity[velocity<0][0]
            err_vel = np.abs(v0)
            tend = t_slip_phase[velocity<0][0]
            t_slip_phase = np.linspace(t_stick_phase[-1],tend,1000)
            w0 = [x[-1],v[-1],press[-1]]
            wsol = odeint(slip_phase, w0, t_slip_phase, args=(p,))
            x = np.concatenate((x,wsol[:,0]))
            dx_slip.append(wsol[-1,0]-wsol[0,0])
            x_stick = x[-1]
            v = np.concatenate((v,wsol[:,1]))
            press = np.concatenate((press,wsol[:,2]))
            pslip.append(wsol[0,2])
            dpslip.append(wsol[-1,2] - wsol[0,2])
            slip_duration.append(t_slip_phase[-1] - t_slip_phase[0])
            t = np.concatenate((t,t_slip_phase))
            t_stick_phase, pres_stick,err_flag = stick_phase(press[-1],t[-1],x[-1],dt,R1,R2,R3,R4,R5)
            press = np.concatenate((press,pres_stick))
            x = np.concatenate((x,x_stick * np.ones(len(t_stick_phase))))
            v = np.concatenate((v,np.zeros(len(t_stick_phase))))
            t  = np.concatenate((t,t_stick_phase))
            q = R2*R3 + R2 * press
            N_cycle = N_cycle + 1
        except ValueError:
            print('Error...')
            err_flag = 1
    dt = np.diff(t)
    V = np.zeros(len(t))
    q = R2*R3 + R2 * press
    Vc = 0
    for i in range(len(dt)):
        dV = q[i]*dt[i]
        Vc = Vc + dV
        V[i+1]= Vc

    return t,press,x,v,V


