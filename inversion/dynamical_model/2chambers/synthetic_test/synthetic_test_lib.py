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

def CalcTilt(DeltaP,xSour,ySour,depth,xSt,ySt,V,pois,lam):
    #Calculate the tilt at one station due to a pressure source which is function of time 
    cs = -9. /(4 * 3.14) * (1 - pois) / lam
    tx = cs * V * DeltaP * depth * (xSt -  xSour)  / (depth**2 + (xSt -  xSour)**2 + (ySt -  ySour)**2 )**(5./2)
    ty = cs * V * DeltaP * depth * (ySt -  ySour)  / (depth**2 + (xSt -  xSour)**2 + (ySt -  ySour)**2 )**(5./2)
    return tx,ty
#def calcGPS():
    
def addNoiseTilt(tiltx,tilty,sigma):
    np.random.seed(42)
    tiltx = tiltx + sigma * np.random.randn(np.shape(tiltx)[0],np.shape(tiltx)[1])
    tilty = tilty + sigma * np.random.randn(np.shape(tiltx)[0],np.shape(tiltx)[1])
    return tiltx,tilty

def Twochambers_syntethic_LF_check(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,mu,pt,Rcyl,rho,g,poisson,lame):
    #This routine compare the timeseries for pressure and tilt obtained by calculating t
    #the time vector with the results obtained by receiving an imposed time vector 
    #(which is the situation for real data). They should be the same, but with different timestep
    #Calculate synthetic timeseries with 2 chambers
    # The input parameters are the one we do not want to invert for, are fixed
    #Source location
    xs = xSource[0]
    ys = ySource[0]
    xd = xSource[1]
    yd = ySource[1]
    depths = depthSource[0]
    depthd = depthSource[1] 
    condd = 4
    conds = 4
    #Chambers parameters
    Vs = 1e+10
    Vd  = 5e+9
    ks = 1e+9
    #Pressures and friction
    taus = 5e+6
    taud = 3e+6
    
    #Cylinder parameters
    S = 3.14 * Rcyl**2 
    #In the following part we calculate tilt 
    ##Calculating parameters
    
    R1 = rho * g * Vd /(ks*S)
    T1 = (conds / condd )**4 * ld /ls
    PHI = ks /ks * Vs / Vd
    R5 = taud / taus
    R3 = pt  / taus
    params = [T1,PHI,R3] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
    un_plus_T1 = 1 + T1
    un_plus_R1 = 1 + R1
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1,R3,R5,T1,PHI,A,B,C,D]
    #Original timseries
    #Initialization
    tcounter = 0
    ps_orig = []
    pd_orig = []
    t_orig = []
    PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
    PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
    PS0 = PD0 
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    TSLIP_seed = 1
    N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
    for i in range(1,N_cycles + 1):
        tseg,tslip,PS,PD,ps0 = TwoChambers_LF(w0,params,PSLIP,TSLIP,TSLIP_seed)
        TSLIP_seed = tslip
        ps_orig.append(PS)
        pd_orig.append(PD)
        t_orig.append(tseg + tcounter)
        PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
        w0 = np.array([PS0,PD0])
        tcounter = tcounter + tslip
    ps_orig = np.concatenate((ps_orig)) * taus
    pd_orig = np.concatenate((pd_orig)) * taus
    t_orig = np.concatenate((t_orig)) * tstar
    
    #Resempling
    t_tilt = np.linspace(0,np.max(t_orig),N_cycles * 200)
    t_gps = np.linspace(0,np.max(t_orig),N_cycles * 1.5)
    t_tilt = t_tilt / tstar
    tcounter = 0
    time1 = time.time()
    ps_data = np.ones(len(t_tilt))
    pd_data = np.ones(len(t_tilt))
    gps_data = np.ones(len(t_gps))
    PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
    PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
    PS0 = -1
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
    for i in range(1,N_cycles + 1):
        tslip,ps_data,pd_data,ps0,gps_data = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,t_tilt,ps_data,pd_data,t_gps,gps_data,i)
        PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
    
    ps_data = ps_data * taus
    pd_data = pd_data * taus
    t_tilt = t_tilt * tstar
    #1 and 2 for the tilts represent the number of the station
    tiltx1 = np.zeros(len(t_tilt))
    tilty1 = np.zeros(len(t_tilt))
    tiltx2 = np.zeros(len(t_tilt))
    tilty2 =  np.zeros(len(t_tilt))
    #Station 1 for two sources
    tempx,tempy = CalcTilt(ps_data,xs,ys, depths, xstation1, ystation1, Vs, poisson, lame)
    tiltx1 = tiltx1 + tempx
    tilty1 = tilty1 + tempy
    tempx,tempy = CalcTilt(pd_data,xd,yd, depthd, xstation1, ystation1, Vd, poisson, lame)
    tiltx1 = tiltx1 + tempx
    tilty1 = tilty1 + tempy
    #Same thing for station 2
    tempx,tempy = CalcTilt(ps_data,xs,ys, depths, xstation2, ystation2, Vs, poisson, lame)
    tiltx2 = tiltx2 + tempx
    tilty2 = tilty2 + tempy
    tempx,tempy = CalcTilt(pd_data,xd,yd, depthd, xstation2, ystation2, Vd, poisson, lame)
    tiltx2 = tiltx2 + tempx
    tilty2 = tilty2 + tempy
    time2 = time.time()
    print('Total time is ',time1 - time2)
    return ps_data,pd_data,t_tilt,ps_orig,pd_orig,tiltx1,tilty1,tiltx2,tilty2,t_orig
    
    
def Twochambers_syntethic_LF(xSource,ySource,depthSource,Xst,Yst,ls,ld,conds,condd,Vs,Vd,ks,taus,taud,mu,pt,Rcyl,rho,g,poisson,lame,tiltsigma,offx,offy):
    #LF = Lower Feeder: in this case the piston is collapsing in the chamber t
    #hat is not feeding directly the eruption (it has only one conduit) 
    #CAREFUL: if you want to change this you have to change the parameters for xstar
    #Calculate synthetic timeseries,including GPS with 2 chambers
    # The input parameters are the one we do not want to invert for, are fixed
    #SOurces locations
    xs = xSource[0]
    ys = ySource[0]
    xd = xSource[1]
    yd = ySource[1]
    ds = depthSource[0]
    dd = depthSource[1] 
    cs = -9. /(4 * 3.14) * (1 - poisson) / lame

    S = 3.14 * Rcyl**2 
    
    R1 = rho * g * Vd /(ks*S)
    T1 = (conds / condd )**4 * ld /ls
    PHI = ks /ks * Vs / Vd
    R5 = taud / taus
    R3 = pt  / taus
    params = [T1,PHI,R3] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
    xstar = taus * Vd / (ks * S) 
    #Careful if you want to change to the UF version (this is LF)
    #xstar should be 
    #xstar = taud * Vs / (ks * S**2)
    un_plus_T1 = 1 + T1
    un_plus_R1 = 1 + R1
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1,R3,R5,T1,PHI,A,B,C,D]
    #Original timseries
    #Initialization
    tcounter = 0
    ps_orig = []
    pd_orig = []
    t_orig = []
    PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
    PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
    TSLIP = 0
    TSLIP_seed = 1
    tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0,0]),params,-1,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 =  ps0
    w0 = np.array([PS0,PD0])
    N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
    for i in range(1,N_cycles + 1):
        tseg,tslip,PS,PD,ps0 = TwoChambers_LF(w0,params,PSLIP,TSLIP,TSLIP_seed)
        TSLIP_seed = tslip
        ps_orig.append(PS)
        pd_orig.append(PD)
        t_orig.append(tseg + tcounter)
        PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
        w0 = np.array([PS0,PD0])
        tcounter = tcounter + tslip

    t_orig = np.concatenate((t_orig)) * tstar
    
    #Resempling
    t_tilt = np.linspace(0,np.max(t_orig),int(N_cycles * 60)) / tstar
    t_gps = np.linspace(0,np.max(t_orig),int(N_cycles * 1.5)) / tstar
    tcounter = 0
    ps_data = np.ones(len(t_tilt))
    pd_data = np.ones(len(t_tilt))
    gps_data = np.ones(len(t_gps))
    PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
    PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
    TSLIP = 0
    TSLIP_seed = 1
    tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0,0]),params,-1,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 =  ps0
    w0 = np.array([PS0,PD0])
    N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
    i  = 1
    flag = 1
    while i < N_cycles + 1 and flag == 1:
        tslip,ps_data,pd_data,ps0,gps_data = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,t_tilt,ps_data,pd_data,t_gps,gps_data,i)
        PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    
    ps = ps_data * taus
    pd = pd_data * taus
    t_tilt = t_tilt * tstar
    gps_data = gps_data * xstar
    t_gps = t_gps * tstar
    
    coeffxs = cs * ds * (Xst -  xs) / (ds**2 + (Xst -  xs)**2 + (Yst -  ys)**2 )**(5./2) * np.ones((len(t_tilt),np.shape(Xst)[0]))
    coeffys = cs * ds * (Yst -  ys) / (ds**2 + (Xst -  xs)**2 + (Yst -  ys)**2 )**(5./2) * np.ones((len(t_tilt),np.shape(Xst)[0]))
    coeffxd = cs * dd * (Xst -  xd) / (dd**2 + (Xst -  xd)**2 + (Yst -  yd)**2 )**(5./2) * np.ones((len(t_tilt),np.shape(Xst)[0]))
    coeffyd = cs * dd * (Yst -  yd) / (dd**2 + (Xst -  xd)**2 + (Yst -  yd)**2 )**(5./2) * np.ones((len(t_tilt),np.shape(Xst)[0]))
    
    tiltx = np.ones(np.shape(coeffxs))
    tilty = np.ones(np.shape(coeffxs))
    tiltxoff = np.ones(np.shape(coeffxs))
    tiltyoff = np.ones(np.shape(coeffxs))
    for i in range(len(Xst)):
        tiltxoff[:,i] = coeffxs[:,i] * Vs * ps + coeffxd[:,i] * Vd * pd + offx[i]*1e-6
        tiltyoff[:,i] = coeffys[:,i] * Vs * ps + coeffyd[:,i] * Vd * pd + offy[i]*1e-6
    
    for i in range(len(Xst)):
        tiltx[:,i] = coeffxs[:,i] * Vs * ps + coeffxd[:,i] * Vd * pd 
        tilty[:,i] = coeffys[:,i] * Vs * ps + coeffyd[:,i] * Vd * pd  
    
    tiltxoff,tiltyoff = addNoiseTilt(tiltxoff, tiltyoff,tiltsigma)
   
    return t_tilt,tiltx,tilty,tiltxoff,tiltyoff,gps_data,t_gps,N_cycles





def DirectModelEmcee_inv(tOrigTilt,tOrigGPS,
                         offx,offy,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                         VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                         Xst,Yst,
                         ls,ld,pt,mu,
                         rhog,cs,S,nstation):


    
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
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod

   