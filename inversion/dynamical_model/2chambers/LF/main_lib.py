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





def DirectModelEmcee_inv_LF(tOrigTilt,tOrigGPS,
                         offtimeSamp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                         VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                         Xst,Yst,
                         ls,ld,pt,mu,
                         rhog,cs,S,nstation):


    tOrigTilt = tOrigTilt + offtimeSamp
    tOrigGPS = tOrigGPS + offtimeSamp
    
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
        txMod[nstation == i] = txMod[nstation == i] + offxSamp[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offySamp[i]*1e-6
    gpsMod = gpsMod  + offGPSSamp
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod


def log_likelihood(param,
                   xstation,ystation,
                   ls,ld,pt,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr,dtxObs,dtyObs,dtiltErr,nstation):
    offtimeSamp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
    offxSamp = np.array([offx1])
    offySamp = np.array([offy1])
    txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              offtimeSamp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              xstation,ystation,
                                              ls,ld,pt,mu,
                                              rhog,const,S,nstation)
    sigma2Tilt = tiltErr ** 2
    sigma2dTilt = dtiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx = -0.5 * np.sum((txObs - txMod) ** 2 / sigma2Tilt,0)  -len(txObs)/ 2 * np.log(6.28 * sigma2Tilt**2)  
    likety = -0.5 * np.sum((tyObs - tyMod) ** 2 / sigma2Tilt,0)  -len(tyObs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    #likedtx = -0.5 * np.sum((dtxObs - dtxMod) ** 2 / sigma2dTilt,0)  -len(dtxObs)/ 2 * np.log(6.28 * sigma2dTilt**2)  
    #likedty = -0.5 * np.sum((dtyObs - dtyMod) ** 2 / sigma2dTilt,0)  -len(dtyObs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    liketilt =  + liketx + likety 
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
     
    return liketilt + likeGPS

def log_prior(param,S,rhog,bounds,bndtimeconst,bndGPSconst,locTr,locEr,Nobs):
   offtimeSamp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
   kSamp = 10**kExpSamp
   R5Samp = 10**R5ExpSamp
   VsSamp= 10**VsExpSamp
   VdSamp = 10**VdExpSamp
   R1Samp = rhog * VdSamp /(kSamp*S)
   offs = np.array([offx1,offy1,])
   
   if bounds[0,0] < VsExpSamp < bounds[0,1] and bounds[1,0] < VdExpSamp < bounds[1,1] and bounds[2,0] < kExpSamp < bounds[2,1] and bounds[3,0] < R5ExpSamp < bounds[3,1] and 2* R1Samp* (Nobs -5) * (1 - R5Samp) / (R1Samp +1) +1   < R3Samp < 2* R1Samp* (Nobs + 30) * (1 - R5Samp) / (R1Samp +1) +1 and bounds[5,0] < condsSamp < bounds[5,1] and bounds[6,0] < conddSamp < bounds[6,1] and all(np.abs(offs)<3e+3) and -bndGPSconst < offGPSSamp < 0    and 0 < offtimeSamp < bndtimeconst:                         
       logprob =   np.log(1.0/(np.sqrt(6.28)*locEr[0]))-0.5*(xsSamp-locTr[0])**2/locEr[0]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[1]))-0.5*(ysSamp-locTr[1])**2/locEr[1]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[2]))-0.5*(dsSamp-locTr[2])**2/locEr[2]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[3]))-0.5*(xdSamp-locTr[3])**2/locEr[3]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[4]))-0.5*(ydSamp-locTr[4])**2/locEr[4]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr[5]))-0.5*(ddSamp-locTr[5])**2/locEr[5]**2
       return logprob
   return -np.inf

def log_probability(param,
                    xstation,ystation,
                    ls,ld,pt,mu,
                    rhog,const,S,
                    tTilt,tGPS,tx,ty,GPS,
                    tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation):
    
    lp = log_prior(param,S,rhog,bounds,bndtimeconst,bndGPSconst,locTruth,locErr,Nmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param,
                               xstation,ystation,
                               ls,ld,pt,mu,
                               rhog,const,S,
                               tTilt,tGPS,tx,ty,GPS,
                               tiltErr,GPSErr,dtx,dty,dtiltErr,nstation)   