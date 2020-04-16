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

def ps_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def pd_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    pd = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return pd 

def pd_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def TwoChambers_UF(w0,par,pslip,tslip,tsl_seed):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    tslip = optimize.brentq(ps_analytic_root,0,1e+7, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tsegment = np.linspace(0,tslip,30)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdend = pd_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tsegment,tslip,pssegment,pdsegment,pdend

def TwoChambers_UF_timein(w0,par,pslip,tslip,time,ps,pd,t_x,x_data,N):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    tsegment = time[time >= tslip]
    
    pssegment = ps_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    ps[time >= tslip] = pssegment
    pd[time >= tslip] = pdsegment
    x_data[t_x >= tslip] = 2 * (1 - r5) / (1 + r1) * N
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(ps_analytic_root,0,1e+12, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    pdend = pd_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tslip,ps,pd,pdend,x_data


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

def DirectModelEmcee_inv_UF(tOrigTilt,tOrigGPS,
                         deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                         VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                         Xst,Yst,
                         ls,ld,mu,
                         rhog,cs,S,nstation):
    VsSamp = 10**VsExpSamp
    VdSamp  = 10**VdExpSamp
    ksSamp = 10**ksExpSamp
    kdSamp = 10**kdExpSamp
    R5Samp =0
    
    R1Samp = rhog * VsSamp /(ksSamp*S)
    T1 = (condsSamp / conddSamp )**4 * ld /ls
    PHI = kdSamp /ksSamp * VsSamp / VdSamp
    tstar = VsSamp * 8 * mu * ld / (ksSamp * 3.14 * conddSamp**4)
    xstar = pspdSamp * VsSamp / (ksSamp * S) 
    deltap0adim = deltap0Samp / pspdSamp
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
    PS0 =   + 2 * (1 - R5Samp) / un_plus_R1 
    PSLIP = - 2 * R1Samp * (1 - R5Samp)/un_plus_R1
    TSLIP = 0
    TSLIP_seed = 1
    #tseg,tslip,PS,PD,pd0 = TwoChambers_UF(np.array([0.1,0.1]),params,0,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PD0 =  PS0 + deltap0adim
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles =  ((1 + R1Samp)/ (2 * R1Samp) * R3Samp)-1
    i  = 1
    thresh = 80
    while i < N_cycles + 1 and i < thresh:
        tslip,ps,pd,pd0,gps = TwoChambers_UF_timein(w0,params,PSLIP,TSLIP,tOrigTilt,ps,pd,tOrigGPS,gps,i)
        PS0 =   + 2 * (1 - R5Samp) / un_plus_R1 -2 * R1Samp * (1 - R5Samp)/un_plus_R1 * i
        PD0 = pd0
        PSLIP =   - 2 * R1Samp * (1 - R5Samp)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    ps = ps * pspdSamp
    pd = pd * pspdSamp
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
        txMod[nstation == i] = txMod[nstation == i] + offxSamp[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offySamp[i]*1e-6
    gpsMod = gpsMod  + offGPSSamp
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod

def DirectModelEmcee_inv_LF(tOrigTilt,tOrigGPS,
                         deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                         VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                         Xst,Yst,
                         ls,ld,mu,
                         rhog,cs,S,nstation):

    VsSamp = 10**VsExpSamp
    VdSamp  = 10**VdExpSamp
    ksSamp = 10**ksExpSamp
    kdSamp = 10**kdExpSamp
    R5Samp =0
    
    R1Samp = rhog * VdSamp /(kdSamp*S)
    T1 = (condsSamp / conddSamp )**4 * ld /ls
    PHI = kdSamp /ksSamp * VsSamp / VdSamp
    params = [T1,PHI,R3Samp] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = VsSamp * 8 * mu * ld / (ksSamp * 3.14 * conddSamp**4)
    xstar = pspdSamp * VdSamp / (kdSamp * S)
    deltap0adim = deltap0Samp / pspdSamp

    #Careful if you want to change to the UF version (this is LF)
    #xstar should be 
    #xstar = taud * Vs / (ks * S**2)
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
    PD0 =   + 2 * (1 - R5Samp) / un_plus_R1 
    PSLIP = - 2 * R1Samp * (1 - R5Samp)/un_plus_R1
    TSLIP = 0
    TSLIP_seed = 1
    #tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0.1,0.1]),params,0,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 =  PD0 + deltap0adim
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles = int(np.ceil((1 + R1Samp)/(2 * R1Samp) * R3Samp)) - 1
    i  = 1
    thresh = 80
    while i < N_cycles + 1 and i < thresh:
        tslip,ps,pd,ps0,gps = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,tOrigTilt,ps,pd,tOrigGPS,gps,i)
        PD0 =   + 2 * (1 - R5Samp) / un_plus_R1 -2 * R1Samp * (1 - R5Samp)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =   - 2 * R1Samp * (1 - R5Samp)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    ps = ps * pspdSamp
    pd = pd * pspdSamp
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
        txMod[nstation == i] = txMod[nstation == i] + offxSamp[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offySamp[i]*1e-6
    gpsMod = gpsMod  + offGPSSamp
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod



def log_likelihood_UF(param,
                   xstation,ystation,
                   ls,ld,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr,nstation):
    if np.max(nstation) ==  0:
        deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1])
        offySamp = np.array([offy1])
    if np.max(nstation) ==  1:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1,offx2])
        offySamp = np.array([offy1,offy2])
    if np.max(nstation) == 2:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1,offx2,offx3])
        offySamp = np.array([offy1,offy2,offy3])
        
    txMod,tyMod,GPSMod = DirectModelEmcee_inv_UF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                                              xstation,ystation,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
    sigma2Tilt = tiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx = -0.5 * np.sum((txObs - txMod) ** 2 / sigma2Tilt,0)  -len(txObs)/ 2 * np.log(6.28 * sigma2Tilt**2)  
    likety = -0.5 * np.sum((tyObs - tyMod) ** 2 / sigma2Tilt,0)  -len(tyObs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    liketilt =  + liketx + likety 
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
     
    return liketilt + likeGPS

def log_likelihood_LF(param,
                   xstation,ystation,
                   ls,ld,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr,nstation):
    if np.max(nstation) ==  0:
        deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1])
        offySamp = np.array([offy1])
    if np.max(nstation) ==  1:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1,offx2])
        offySamp = np.array([offy1,offy2])
    if np.max(nstation) == 2:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offxSamp = np.array([offx1,offx2,offx3])
        offySamp = np.array([offy1,offy2,offy3])
        
    txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                                              xstation,ystation,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
    sigma2Tilt = tiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx = -0.5 * np.sum((txObs - txMod) ** 2 / sigma2Tilt,0)  -len(txObs)/ 2 * np.log(6.28 * sigma2Tilt**2)  
    likety = -0.5 * np.sum((tyObs - tyMod) ** 2 / sigma2Tilt,0)  -len(tyObs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    liketilt =  + liketx + likety 
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
     
    return liketilt + likeGPS

def log_prior_UF(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTr,locEr,nstation,flaglocation):
    if np.max(nstation) ==  0:
        deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,])
    if np.max(nstation) ==  1:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,offx2,offy2])
    if np.max(nstation) == 2:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,offx2,offy2,offx3,offy3])
    ksSamp = 10**ksExpSamp
    kdSamp = 10**kdExpSamp
    VsSamp= 10**VsExpSamp
    VdSamp = 10**VdExpSamp
    R1Samp = rhog * VsSamp /(ksSamp*S)
    if flaglocation =='N': # Normal priots
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < VdExpSamp < bounds[1,1])
        conditions.append(bounds[2,0] < ksExpSamp < bounds[2,1])
        conditions.append(bounds[3,0] < kdExpSamp < bounds[3,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[4,0] / (2 * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[4,1] / (2 * R1Samp))
        conditions.append(bounds[5,0] * 2 * R1Samp / (1 + R1Samp) < R3Samp < bounds[5,1] * 2 * R1Samp / (1 + R1Samp))
        conditions.append(bounds[6,0] < condsSamp < bounds[6,1])
        conditions.append(bounds[7,0] < conddSamp < bounds[7,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < bndGPSconst)
        conditions.append(-bndp0 < deltap0Samp < 0)
        if all(conditions):
            logprob =   np.log(1.0/(np.sqrt(6.28)*locEr[0]))-0.5*(xsSamp-locTr[0])**2/locEr[0]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[1]))-0.5*(ysSamp-locTr[1])**2/locEr[1]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[2]))-0.5*(dsSamp-locTr[2])**2/locEr[2]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[3]))-0.5*(xdSamp-locTr[3])**2/locEr[3]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[4]))-0.5*(ydSamp-locTr[4])**2/locEr[4]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[5]))-0.5*(ddSamp-locTr[5])**2/locEr[5]**2
            return logprob
        return -np.inf
    elif flaglocation == 'F': #flat uniform priors
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < VdExpSamp < bounds[1,1])
        conditions.append(bounds[2,0] < ksExpSamp < bounds[2,1])
        conditions.append(bounds[3,0] < kdExpSamp < bounds[3,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[4,0] / (2 * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[4,1] / (2 * R1Samp))
        conditions.append(bounds[5,0] * 2 * R1Samp / (1 + R1Samp) < R3Samp < bounds[5,1] * 2 * R1Samp / (1 + R1Samp))
        conditions.append(bounds[6,0] < condsSamp < bounds[6,1])
        conditions.append(bounds[7,0] < conddSamp < bounds[7,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < bndGPSconst)
        conditions.append(-bndp0 < deltap0Samp < 0)
        conditions.append(boundsLoc[0][0]< xsSamp < boundsLoc[0][1])
        conditions.append(boundsLoc[1][0]< ysSamp < boundsLoc[1][1])
        conditions.append(boundsLoc[2][0]< dsSamp < boundsLoc[2][1])
        conditions.append(boundsLoc[3][0]< xdSamp < boundsLoc[3][1])
        conditions.append(boundsLoc[4][0]< ydSamp < boundsLoc[4][1])
        conditions.append(boundsLoc[5][0]< ddSamp < boundsLoc[5][1])
        if all (conditions):
            return 0
        return -np.inf    

def log_prior_LF(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTr,locEr,nstation,flaglocation):
    if np.max(nstation) ==  0:
        deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,])
    if np.max(nstation) ==  1:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,offx2,offy2])
    if np.max(nstation) == 2:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp, conddSamp = param
        offs = np.array([offx1,offy1,offx2,offy2,offx3,offy3])
    ksSamp = 10**ksExpSamp
    kdSamp = 10**kdExpSamp
    VsSamp= 10**VsExpSamp
    VdSamp = 10**VdExpSamp
    R1Samp = rhog * VdSamp /(kdSamp*S)
    if flaglocation =='N': # Normal priots
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < VdExpSamp < bounds[1,1])
        conditions.append(bounds[2,0] < ksExpSamp < bounds[2,1])
        conditions.append(bounds[3,0] < kdExpSamp < bounds[3,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[4,0] / (2 * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[4,1] / (2 * R1Samp))
        conditions.append(bounds[5,0] * 2 * R1Samp / (1 + R1Samp) < R3Samp < bounds[5,1] * 2 * R1Samp / (1 + R1Samp))
        conditions.append(bounds[6,0] < condsSamp < bounds[6,1])
        conditions.append(bounds[7,0] < conddSamp < bounds[7,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < bndGPSconst)
        conditions.append(-bndp0 < deltap0Samp < 0)
        if all(conditions):
            logprob =   np.log(1.0/(np.sqrt(6.28)*locEr[0]))-0.5*(xsSamp-locTr[0])**2/locEr[0]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[1]))-0.5*(ysSamp-locTr[1])**2/locEr[1]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[2]))-0.5*(dsSamp-locTr[2])**2/locEr[2]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[3]))-0.5*(xdSamp-locTr[3])**2/locEr[3]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[4]))-0.5*(ydSamp-locTr[4])**2/locEr[4]**2
            logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[5]))-0.5*(ddSamp-locTr[5])**2/locEr[5]**2
            return logprob
        return -np.inf
    elif flaglocation == 'F': #flat uniform priors
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < VdExpSamp < bounds[1,1])
        conditions.append(bounds[2,0] < ksExpSamp < bounds[2,1])
        conditions.append(bounds[3,0] < kdExpSamp < bounds[3,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[4,0] / (2 * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[4,1] / (2 * R1Samp))
        conditions.append(bounds[5,0] * 2 * R1Samp / (1 + R1Samp) < R3Samp < bounds[5,1] * 2 * R1Samp / (1 + R1Samp))
        conditions.append(bounds[6,0] < condsSamp < bounds[6,1])
        conditions.append(bounds[7,0] < conddSamp < bounds[7,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < bndGPSconst)
        conditions.append(-bndp0 < deltap0Samp < 0)
        conditions.append(boundsLoc[0][0]< xsSamp < boundsLoc[0][1])
        conditions.append(boundsLoc[1][0]< ysSamp < boundsLoc[1][1])
        conditions.append(boundsLoc[2][0]< dsSamp < boundsLoc[2][1])
        conditions.append(boundsLoc[3][0]< xdSamp < boundsLoc[3][1])
        conditions.append(boundsLoc[4][0]< ydSamp < boundsLoc[4][1])
        conditions.append(boundsLoc[5][0]< ddSamp < boundsLoc[5][1])
        if all (conditions):
            return 0
        return -np.inf
   
def log_probability_UF(param,
                    xstation,ystation,
                    ls,ld,mu,
                    rhog,const,S,
                    tTilt,tGPS,tx,ty,GPS,
                    tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation):
    
    lp = log_prior_UF(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_UF(param,
                               xstation,ystation,
                               ls,ld,mu,
                               rhog,const,S,
                               tTilt,tGPS,tx,ty,GPS,
                               tiltErr,GPSErr,nstation)

    
def log_probability_LF(param,
                    xstation,ystation,
                    ls,ld,mu,
                    rhog,const,S,
                    tTilt,tGPS,tx,ty,GPS,
                    tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation):
    
    lp = log_prior_LF(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_LF(param,
                               xstation,ystation,
                               ls,ld,mu,
                               rhog,const,S,
                               tTilt,tGPS,tx,ty,GPS,
                               tiltErr,GPSErr,nstation)   

def walkers_init(nwalkers,ndim,bounds,boundsLoc,rhog,S,locTruth,locErr,bndtiltconst,bndGPSconst,bndp0,Nst,mt,flaglocation):
    pos = np.zeros((nwalkers*10,ndim)) 
    initial_rand = np.ones(ndim)
    for i in range(len(bounds)):
        pos[:,i] = np.random.uniform(low = bounds[i][0],high = bounds[i][1],size = nwalkers * 10)
    if mt == 'UF':
        R1Initial = rhog * 10**pos[:,0] /(10**pos[:,2] * S)
    elif mt == 'LF':
        R1Initial = rhog * 10**pos[:,1] /(10**pos[:,3] * S)
    lower =  rhog * (1 + R1Initial) * bounds[4,0] / (2 * R1Initial)
    upper =  rhog * (1 + R1Initial) * bounds[4,1] / (2 * R1Initial)
    pos[:,4] = np.random.uniform(low = lower,high = upper, size = nwalkers * 10)
    ind = pos[:,4] < 3e+7
    R1Initial = R1Initial[ind]
    pos = pos[ind,:]
    ind = np.random.uniform(len(pos),size = nwalkers)
    ind = ind.astype(int)
    pos = pos[ind,:]
    R1Initial = R1Initial[ind]
    lower = bounds[5,0] * 2 * R1Initial / (1 + R1Initial)
    upper = bounds[5,1] * 2 * R1Initial / (1 + R1Initial)
    pos[:,5] = np.random.uniform(low = lower,high = upper, size = nwalkers)
    
    locs = np.zeros((nwalkers,6))
    for i in range(len(locTruth)):
        if flaglocation == 'N': # normal priors
            locs[:,i] = np.random.normal(loc = locTruth[i], scale =locErr[i],size = nwalkers)
        elif flaglocation == 'F': #flat uniform priors:
            locs[:,i] = np.random.uniform(low = boundsLoc[i][0] , high = boundsLoc[i][1], size = nwalkers)
    pos = np.concatenate((locs,pos),axis = 1)
    
    offtilt = np.zeros((nwalkers, Nst * 2))
    for i in range(Nst * 2):
        offtilt[:,i] = np.random.uniform(low = -bndtiltconst, high = bndtiltconst, size =nwalkers)
    pos = np.concatenate((offtilt,pos),axis = 1)
    
    offsGPS = np.random.uniform(low = -bndGPSconst , high = 0,size = (nwalkers,1))
    pos = np.concatenate((offsGPS,pos),axis = 1 )
    if mt == 'UF':
        initialp = np.random.uniform(low = 0 , high = bndp0,size = (nwalkers,1))
    elif mt == 'LF':
        initialp = np.random.uniform(low = - bndp0 , high = 0,size = (nwalkers,1))

    pos = np.concatenate((initialp,pos),axis = 1)
    nwalkers, ndim = pos.shape
    
    return pos,nwalkers,ndim
