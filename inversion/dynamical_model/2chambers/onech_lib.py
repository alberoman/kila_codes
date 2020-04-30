#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:35:16 2020

@author: aroman
"""

import numpy as np

def TwoChambers_disc_timein(w0,par,pslip,tslipc,time,ps,t_x,x_data,N,alpha):
    ps0 = w0
    r1,r3 = par
    tsegment = time[time >= tslipc] - tslipc
    
    pssegment = (ps0 + r3) * np.exp(-tsegment) - r3
    ps[time >= tslipc]= pssegment
    x_data[t_x >= tslipc] = 4 * alpha / (1 + r1) * N
    tslip = -np.log((pslip + r3)/(ps0 + r3))
    return tslip,ps,x_data




def DirectModelEmcee_inv_onech(tOrigTilt,tOrigGPS,
                         offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                         VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp,
                         Xst,Yst,
                         ls,mu,
                         rhog,cs,S,nstation):
                         
    VsSamp = 10**VsExpSamp
    ksSamp = 10**ksExpSamp
    R5Samp =0
    
    R1Samp = rhog * VsSamp /(ksSamp*S)

    tstar = VsSamp * 8 * mu * ls / (ksSamp * 3.14 * condsSamp**4)
    xstar = pspdSamp * VsSamp / (ksSamp * S) 
    tOrigGPS = tOrigGPS /tstar
    tOrigTilt = tOrigTilt / tstar
    params = [R1Samp,R3Samp]
    ps = np.ones(len(tOrigTilt))
    gps = np.ones(len(tOrigGPS))
    ps0 = + 4 * alphaSamp / (1 + R1Samp)
    PSLIP = -4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp)
    TSLIPcum = 0
    N_cycles =  ((1 + R1Samp)/ (4 * alphaSamp * R1Samp) * R3Samp)-1
    i  = 1
    w0 = [ps0]
    thresh = 70
    while i < N_cycles + 1 and i < thresh:
        tslip,ps,gps = TwoChambers_disc_timein(w0,params,PSLIP,TSLIPcum,tOrigTilt,ps,tOrigGPS,gps,i,alphaSamp)
        ps0 =     + 4 * alphaSamp / (1 + R1Samp) -4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp) * i
        PSLIP =   - 4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp) * (i + 1)
        TSLIPcum = TSLIPcum + tslip
        w0 = np.array([ps0])
        i = i + 1
    ps = ps * pspdSamp
    coeffxs = cs * dsSamp * (Xst -  xsSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    coeffys = cs * dsSamp * (Yst -  ysSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    txMod = coeffxs * VsSamp * ps 
    tyMod = coeffys * VsSamp * ps 
    gpsMod = gps * xstar
    tOrigGPS = tOrigGPS * tstar
    for i in range((np.max(nstation) + 1)):
        txMod[nstation == i] = txMod[nstation == i] + offxSamp[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offySamp[i]*1e-6
    gpsMod = gpsMod  + offGPSSamp
    return txMod,tyMod,gpsMod#,dtxMod, dtyMod


def DirectModelEmcee_inv_onech_diagno(tOrigTilt,tOrigGPS,
                         offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                         VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp,
                         Xst,Yst,
                         ls,mu,
                         rhog,cs,S,nstation):
                         
    VsSamp = 10**VsExpSamp
    ksSamp = 10**ksExpSamp
    R5Samp =0
    
    R1Samp = rhog * VsSamp /(ksSamp*S)

    tstar = VsSamp * 8 * mu * ls / (ksSamp * 3.14 * condsSamp**4)
    xstar = pspdSamp * VsSamp / (ksSamp * S) 
    tOrigGPS = tOrigGPS /tstar
    tOrigTilt = tOrigTilt / tstar
    params = [R1Samp,R3Samp]
    ps = np.ones(len(tOrigTilt))
    gps = np.ones(len(tOrigGPS))
    ps0 = + 4 * alphaSamp / (1 + R1Samp)
    PSLIP = -4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp)
    TSLIPcum = 0
    N_cycles =  ((1 + R1Samp)/ (4 * alphaSamp * R1Samp) * R3Samp)-1
    i  = 1
    w0 = [ps0]
    thresh = 70
    while i < N_cycles + 1 and i < thresh:
        tslip,ps,gps = TwoChambers_disc_timein(w0,params,PSLIP,TSLIPcum,tOrigTilt,ps,tOrigGPS,gps,i,alphaSamp)
        ps0 =     + 4 * alphaSamp / (1 + R1Samp) -4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp) * i
        PSLIP =   - 4 * alphaSamp * R1Samp * (1 - R5Samp)/(1 + R1Samp) * (i + 1)
        TSLIPcum = TSLIPcum + tslip
        w0 = np.array([ps0])
        i = i + 1
    ps = ps * pspdSamp
    coeffxs = cs * dsSamp * (Xst -  xsSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    coeffys = cs * dsSamp * (Yst -  ysSamp) / (dsSamp**2 + (Xst -  xsSamp)**2 + (Yst -  ysSamp)**2 )**(5./2) 
    txMod = coeffxs * VsSamp * ps 
    tyMod = coeffys * VsSamp * ps 
    gpsMod = gps * xstar
    tOrigGPS = tOrigGPS * tstar
    for i in range((np.max(nstation) + 1)):
        txMod[nstation == i] = txMod[nstation == i] + offxSamp[i]*1e-6
        tyMod[nstation == i] = tyMod[nstation == i] + offySamp[i]*1e-6
    gpsMod = gpsMod  + offGPSSamp
    return txMod,tyMod,gpsMod,ps#,dtxMod, dtyMod


def log_likelihood_onech(param,
                   xstation,ystation,
                   ls,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr,nstation):
    if np.max(nstation) ==  0:
        offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offxSamp = np.array([offx1])
        offySamp = np.array([offy1])
    if np.max(nstation) ==  1:
        offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offxSamp = np.array([offx1,offx2])
        offySamp = np.array([offy1,offy2])
    if np.max(nstation) == 2:
        offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offxSamp = np.array([offx1,offx2,offx3])
        offySamp = np.array([offy1,offy2,offy3])
        
    txMod,tyMod,GPSMod = DirectModelEmcee_inv_onech(tTilt,tGPS,
                                              offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                                              VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp,
                                              xstation,ystation,
                                              ls,mu,
                                              rhog,const,S,nstation)
    sigma2Tilt = tiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx = -0.5 * np.sum((txObs - txMod) ** 2 / sigma2Tilt,0)  -len(txObs)/ 2 * np.log(6.28 * sigma2Tilt)  
    likety = -0.5 * np.sum((tyObs - tyMod) ** 2 / sigma2Tilt,0)  -len(tyObs)/ 2 * np.log(6.28 * sigma2Tilt)
    liketilt =  + liketx + likety 
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS)
     
    return liketilt + likeGPS




def log_prior_onech(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,locTr,locEr,nstation,flaglocation):
    if np.max(nstation) ==  0:
        offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offs = np.array([offx1,offy1,])
    if np.max(nstation) ==  1:
        offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offs = np.array([offx1,offy1,offx2,offy2])
    if np.max(nstation) == 2:
        offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,pspdSamp,R3Samp,condsSamp,alphaSamp = param
        offs = np.array([offx1,offy1,offx2,offy2,offx3,offy3])
    ksSamp = 10**ksExpSamp
    VsSamp= 10**VsExpSamp
    R1Samp = rhog * VsSamp /(ksSamp*S)
    if flaglocation =='N': # Normal priots
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < ksExpSamp < bounds[1,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[2,0] / (4  * alphaSamp * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[2,1] / (4 * alphaSamp * R1Samp))
        conditions.append(bounds[3,0] * 4 * alphaSamp * R1Samp / (1 + R1Samp) < R3Samp < bounds[3,1] * 4 * alphaSamp * R1Samp / (1 + R1Samp))
        conditions.append(bounds[4,0] < condsSamp < bounds[4,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < bndGPSconst)
        if all(conditions):
            logprob =   -0.5 * np.log(6.28 *locEr[0]**2) -0.5*(xsSamp-locTr[0])**2/locEr[0]**2
            logprob = logprob -0.5  * np.log(6.28 *locEr[1]**2) -0.5*(ysSamp-locTr[1])**2/locEr[1]**2
            logprob = logprob -0.5  * np.log(6.28*locEr[2]**2) -0.5*(dsSamp-locTr[2])**2/locEr[2]**2

            return logprob
        return -np.inf
    elif flaglocation == 'F': #flat uniform priors
        conditions = []
        conditions.append(bounds[0,0] < VsExpSamp < bounds[0,1])
        conditions.append(bounds[1,0] < VdExpSamp < bounds[1,1])
        conditions.append(bounds[2,0] < ksExpSamp < bounds[2,1])
        conditions.append(bounds[3,0] < kdExpSamp < bounds[3,1])
        conditions.append(rhog * (1 + R1Samp) * bounds[4,0] / (4 * alphaSamp * R1Samp) < pspdSamp < rhog * (1 + R1Samp) * bounds[4,1] / (4 * alphaSamp * R1Samp))
        conditions.append(bounds[5,0] * 4 * alphaSamp * R1Samp / (1 + R1Samp) < R3Samp < bounds[5,1] * 4 * alphaSamp * R1Samp / (1 + R1Samp))
        conditions.append(bounds[6,0] < condsSamp < bounds[6,1])
        conditions.append(bounds[7,0] < conddSamp < bounds[7,1])
        conditions.append(all(np.abs(offs)<bndtiltconst))
        conditions.append(-bndGPSconst < offGPSSamp < 0)
        conditions.append(boundsLoc[0][0]< xsSamp < boundsLoc[0][1])
        conditions.append(boundsLoc[1][0]< ysSamp < boundsLoc[1][1])
        conditions.append(boundsLoc[2][0]< dsSamp < boundsLoc[2][1])
        conditions.append(boundsLoc[3][0]< xdSamp < boundsLoc[3][1])
        conditions.append(boundsLoc[4][0]< ydSamp < boundsLoc[4][1])
        conditions.append(boundsLoc[5][0]< ddSamp < boundsLoc[5][1])
        if all (conditions):
            return 0
        return -np.inf    


   

def log_probability_onech(param,
                    xstation,ystation,
                    ls,mu,
                    rhog,const,S,
                    tTilt,tGPS,tx,ty,GPS,
                    tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,locTruth,locErr,nstation,flaglocation):
    
    lp = log_prior_onech(param,S,rhog,bounds,boundsLoc,bndGPSconst,bndtiltconst,locTruth,locErr,nstation,flaglocation)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_onech(param,
                               xstation,ystation,
                               ls,mu,
                               rhog,const,S,
                               tTilt,tGPS,tx,ty,GPS,
                               tiltErr,GPSErr,nstation)   

def walkers_init(nwalkers,ndim,bounds,boundsLoc,rhog,S,locTruth,locErr,bndtiltconst,bndGPSconst,Nst,mt,flaglocation):
    pos = np.zeros((nwalkers*10,ndim)) 
    initial_rand = np.ones(ndim)
    for i in range(len(bounds)):
        pos[:,i] = np.random.uniform(low = bounds[i][0],high = bounds[i][1],size = nwalkers * 10)
    if mt == 'UF':
        R1Initial = rhog * 10**pos[:,0] /(10**pos[:,1] * S)
    elif mt == 'LF':
        R1Initial = rhog * 10**pos[:,0] /(10**pos[:,1] * S)
    lower =  rhog * (1 + R1Initial) * bounds[2,0] / (4 * pos[:,-1] * R1Initial)
    upper =  rhog * (1 + R1Initial) * bounds[2,1] / (4 * pos[:,-1] * R1Initial)
    pos[:,2] = np.random.uniform(low = lower,high = upper, size = nwalkers * 10)
    ind = pos[:,2] < 3e+7
    R1Initial = R1Initial[ind]
    pos = pos[ind,:]
    ind = np.random.uniform(len(pos),size = nwalkers)
    ind = ind.astype(int)
    pos = pos[ind,:]
    R1Initial = R1Initial[ind]
    lower = bounds[3,0] * 4 * pos[:,-1] * R1Initial / (1 + R1Initial)
    upper = bounds[3,1] * 4 * pos[:,-1] * R1Initial / (1 + R1Initial)
    pos[:,3] = np.random.uniform(low = lower,high = upper, size = nwalkers)
    
    locs = np.zeros((nwalkers,3))
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
    
    nwalkers, ndim = pos.shape
    
    return pos,nwalkers,ndim
