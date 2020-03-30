#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:35:16 2020

@author: aroman
"""

import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
import time 

def ps_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    ps = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return  ps
def ps_diff(ps,pd,R3,T1):
    return  pd - ps - T1 * (ps + R3)

def ps_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def pd_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    pd = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return pd
def pd_diff(ps,pd, phi):
    return - phi * (pd - ps) 

def pd_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def TwoChambers_UF(w0,par,pslip,tslip,tsl_seed):
    ps0,pd0 = w0
    r3,t1,phi,a,b,c,d = par
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(ps_analytic_root,0,1e+7, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tsegment = np.linspace(0,tslip,30)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdend = pd_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tsegment,tslip,pssegment,pdsegment,pdend

def TwoChambers_LF(w0,par,pslip,tslip,tsl_seed):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(pd_analytic_root,0,1e+9, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tsegment = np.linspace(0,tslip,50)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    psend = pssegment[-1]
    return tsegment,tslip,pssegment,pdsegment,psend

def TwoChambers_LF_timein(w0,par,pslip,tslip,time,ps,pd,t_x,x_data,N):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    tsegment = time[time >= tslip]
    
    pssegment = ps_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    ps[time >= tslip] = pssegment
    pd[time >= tslip] = pdsegment
    x_data[t_x >= tslip] = 2 * (1 - r5) / (1 + r1) * N
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(pd_analytic_root,0,1e+8, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    psend = ps_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tslip,ps,pd,psend,x_data





def DirectModelEmcee_inv(tOrigTilt,tOrigGPS,
                         offx,offy,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                         VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                         Xst,Yst,
                         ls,ld,pt,mu,
                         rhog,cs,S,nstation):


    #tOrigTilt = tOrigTilt #+ offtimeSamp
    #tOrigGPS = tOrigGPS #+ offtimeSamp
    
    VsSamp = 10**VsExpSamp
    VdSamp  = 10**VdExpSamp
    ksSamp = 10**kExpSamp

    R5Samp = 10**R5ExpSamp
    tausSamp = pt / R3Samp
    
    R1Samp = rhog * VdSamp /(ksSamp*S)
    T1 = (condsSamp / conddSamp )**4 * ld /ls
    PHI = ksSamp /ksSamp * VsSamp / VdSamp
    params = [T1,PHI,R3Samp] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = VsSamp * 8 * mu * ld / (ksSamp * 3.14 * conddSamp**4)
    xstar = tausSamp * VdSamp / (ksSamp * S) 
    #Careful if you want to change to the UF version (this is LF)
    #xstar should be 
    #xstar = taud * Vs / (ks * S**2)
    un_plus_T1 = 1 + T1
    un_plus_R1 = 1 + R1Samp
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1Samp,R3Samp,R5Samp,T1,PHI,A,B,C,D]
   
    
    
    tOrigGPS = tOrigGPS /tstar
    tOrigTilt = tOrigTilt / tstar
    ps = np.ones(len(tOrigTilt))
    pd = np.ones(len(tOrigTilt))
    gps = np.ones(len(tOrigGPS))
    PD0 =  - 1  + 2 * (1 - R5Samp) / un_plus_R1 
    PSLIP = - 1 - 2 * R1Samp * (1 - R5Samp)/un_plus_R1
    TSLIP = 0
    TSLIP_seed = 1
    tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0,0]),params,-1,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 =  ps0    
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles = int(np.ceil((R3Samp - 1) * (R1Samp + 1) / (2 * (1 - R5Samp) * R1Samp))) - 1
    i  = 1
    while i < N_cycles + 1:
        tslip,ps,pd,ps0,gps = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,tOrigTilt,ps,pd,tOrigGPS,gps,i)
        PD0 = - 1  + 2 * (1 - R5Samp) / un_plus_R1 -2 * R1Samp * (1 - R5Samp)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1Samp * (1 - R5Samp)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    ps = ps * tausSamp
    pd = pd * tausSamp
    coeffxs = cs * dsSamp * (Xst -  xsSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    coeffys = cs * dsSamp * (Yst -  ysSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    coeffxd = cs * ddSamp * (Xst -  xdSamp) / (ddSamp**2 + (Xst -  xdSamp)**2 + (Yst -  ydSamp)**2 )**(5./2) 
    coeffyd = cs * ddSamp * (Yst -  ydSamp) / (ddSamp**2 + (Xst -  xdSamp)**2 + (Yst -  ydSamp)**2 )**(5./2) 
    txMod = coeffxs * VsSamp * ps + coeffxd * VdSamp * pd
    tyMod = coeffys * VsSamp * ps + coeffyd * VdSamp * pd
    #dtxMod = np.diff(txMod[j]) / np.diff(tTiltList[j]))
    #dtyMod.append(np.diff(tyMod[j]) / np.diff(tTiltList[j]))
    
    gpsMod = gps * xstar
    tOrigTilt = tOrigTilt / tstar
    tOrigGPS = tOrigGPS * tstar
    for i in range((np.max(nstation) + 1)):
        txMod[nstation == i] = txMod[nstation == i] + offx[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offy[i]*1e-6
    gpsMod = gpsMod 
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod

   