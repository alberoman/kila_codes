#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
"""

import matplotlib.pyplot as plt
import numpy as np
from main_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from multiprocessing import Pool
import os

os.environ["OMP_NUM_THREADS"] = "1" 

def log_likelihood(param,
                   xstation,ystation,
                   ls,ld,pt,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr,dtxObs,dtyObs,dtiltErr,nstation):
    offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
    offxSamp = np.array([offx1])
    offySamp = np.array([offy1])
    txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                                              offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
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

def log_prior(param,S,rhog,bounds,locTr,locEr,Nobs):
   offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
   kSamp = 10**kExpSamp
   R5Samp = 10**R5ExpSamp
   VsSamp= 10**VsExpSamp
   VdSamp = 10**VdExpSamp
   R1Samp = rhog * VdSamp /(kSamp*S)
   offs = np.array([offx1,offy1])
   
   if bounds[0,0] < VsExpSamp < bounds[0,1] and bounds[1,0] < VdExpSamp < bounds[1,1] and bounds[2,0] < kExpSamp < bounds[2,1] and bounds[3,0] < R5ExpSamp < bounds[3,1] and 2* R1Samp* (Nobs -5) * (1 - R5Samp) / (R1Samp +1) +1   < R3Samp < 2* R1Samp* (Nobs + 15) * (1 - R5Samp) / (R1Samp +1) +1 and bounds[5,0] < condsSamp < bounds[5,1] and bounds[6,0] < conddSamp < bounds[6,1] and all(np.abs(offs)<3e+3):                             
       logprob =   np.log(1.0/(np.sqrt(6.28)*locErr[0]))-0.5*(xsSamp-locTr[0])**2/locErr[0]**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locErr[1]))-0.5*(ysSamp-locTr[1])**2/locEr[1]**2
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
                    tiltErr,GPSErr,bounds,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation):
    
    lp = log_prior(param,S,rhog,bounds,locTruth,locErr,Nmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param,
                               xstation,ystation,
                               ls,ld,pt,mu,
                               rhog,const,S,
                               tTilt,tGPS,tx,ty,GPS,
                               tiltErr,GPSErr,dtx,dty,dtiltErr,nstation)
np.random.seed(1234)

roffsetx = np.array([700,-400,150])
offsety = np.array([300,-1020,-10])
xSource = np.array([0,0]) #The first value here corresponds to the shallow source (the one feeding)
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = -2000
ystation2 = -4000
xstation3 = 3000
ystation3 = -2000
sigma_tilt = 0.2
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 3e+4
ld = 2.5e+3
condd = 4
conds = 4
#Sources volumes and compressibilities
Vd_exp = 9.698970004336019
Vs_exp = 11
k_exp = 9
Vd = 10**Vd_exp
Vs = 10**Vs_exp
ks = 10**k_exp
#Pressures and friction6
R3 = 3
R5 = 0.8
R5exp = np.log10(R5)
taus = 7e+6
taud = taus * R5
mu = 100
#Pressures and friction
pt = taus * R3
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 7.0e+2
S = 3.14 * Rcyl**2 
const = -9. /(4 * 3.14) * (1 - poisson) / lame

tiltPert = 2e-5
tiltErr = 1e-4
GPSErr = 1e-1
locErr = 500
startstation3 =400
xsh = xSource[0]
xde = xSource[1]
ysh = ySource[0]
yde = ySource[1]
dsh = depthSource[0]
dde = depthSource[1]
nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS.pickle','rb'))
Nmax = 20
xsh = locTruth[0]
ysh = locTruth[1]
dsh = locTruth[2]
xde = locTruth[3]
yde = locTruth[4]
dde = locTruth[5]
offx =  np.array([0])
offy =  np.array([0])

txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                        offx,offy,xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
                        rhog,const,S,nstation)