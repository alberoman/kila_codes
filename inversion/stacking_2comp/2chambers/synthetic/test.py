#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:52:31 2020

@author: aroman
"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from multiprocessing import Pool
import os
def log_likelihood(param, xS,yS,depthS,posXst1,posYst1,posXst2,posYst2,
                               lonconds,loncondd,pressureTopo,viscosity,
                               densGrav,elastic,Scyl,tTiltmeter,timePiston,
                               tx1Obs,ty1Obs,tx2Obs,ty2Obs,GPSObs,
                               tiltEpsi,GPSEpsi):
    
    VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
    tx1Mod,ty1Mod,tx2Mod,ty2Mod,GPSMod = DirectModelEmcee_inv(tTiltmeter,timePiston,
                                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                                              xS,yS,depthS,posXst1,posYst1,posXst2,posYst2,
                                                              lonconds,loncondd,pressureTopo,viscosity,
                                                              densGrav,elastic,Scyl)
    sigma2Tilt = tiltEpsi ** 2
    sigma2GPS = GPSEpsi ** 2
    liketx1 = -0.5 * np.sum((tx1Obs - tx1Mod) ** 2 / sigma2Tilt )  -len(tx1Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    likety1 = -0.5 * np.sum((ty1Obs - ty1Mod) ** 2 / sigma2Tilt )  -len(ty1Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    liketx2 = -0.5 * np.sum((tx2Obs - tx2Mod) ** 2 / sigma2Tilt )  -len(tx2Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    likety2 = -0.5 * np.sum((ty2Obs - ty2Mod) ** 2 / sigma2Tilt )  -len(ty2Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    
    #likedtx1 = -0.5 * np.sum((dtx1Obs - dtx1Mod) ** 2 / sigma2dTilt ) -len(dtx1Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    #likedty1 = -0.5 * np.sum((dty1Obs - dty1Mod) ** 2 / sigma2dTilt ) -len(dty1Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    #likedtx2 = -0.5 * np.sum((dtx2Obs - dtx2Mod) ** 2 / sigma2dTilt )-len(dtx2Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    #likedty2 = -0.5 * np.sum((dty2Obs - dty2Mod) ** 2 / sigma2dTilt ) -len(dty2Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
    return +liketx1  + likety1 + liketx2 + likety2+ likeGPS# +likedtx1 +likety1 +likedtx2 +likedty2 

def log_prior(parameters,SCyl,DensGrav,bounds,Nobs):
   VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
   kSamp = 10**kExpSamp
   R5Samp = 10**R5ExpSamp
   VsSamp= 10**VsExpSamp
   VdSamp = 10**VdExpSamp
   R1Samp = DensGrav * VdSamp /(kSamp*SCyl)
   if bounds[0,0] < VsSamp < bounds[0,1] and bounds[1,0] < VdSamp < bounds[1,1] and bounds[2,0] < kSamp < bounds[2,1] and bounds[3,0] < R5Samp < bounds[3,1] and 2* R1Samp* Nobs * (1 - R5Samp) / (R1Samp +1) +1   < R3Samp < 2* R1Samp* (Nobs +10) * (1 - R5Samp) / (R1Samp +1) +1 and bounds[5,0] < conds < bounds[5,1] and bounds[6,0] < condd < bounds[6,1]:
       return 0
   return -np.inf

def log_probability(param,xSo,ySo,depthSo,xst1,yst1,xst2,yst2,
                    loncs,loncd,ptopo,visco,
                    rhog,elasticC,surf,
                    timeTilt,timeGPS,tiltx1,tilty1,tiltx2,tilty2,position,tiltEpsilon,GPSEpsilon,priors,N):
    lp = log_prior(param,surf,rhog,priors,N)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param,xSo,ySo,depthSo,xst1,yst1,xst2,yst2,
                               loncs,loncd,ptopo,visco,
                               rhog,elasticC,surf,timeTilt,timeGPS,
                               tiltx1,tilty1,tiltx2,tilty2,position,
                               tiltEpsilon,GPSEpsilon)
  

xSource = np.array([0,0]) #The first value here corresponds to the shallow source (the one feeding)
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = 100
ystation2 = -4000
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
Vs_exp = 10
k_exp = 9
Vd = 10**Vd_exp
Vs = 10**Vs_exp
ks = 10**k_exp
#Pressures and friction6
R3 = 3
R5 = 0.2
R5exp = np.log10(R5)
taus = 7e+6
taud = taus * R5
mu = 100
#Pressures and friction
pt = taus * R3
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
tiltPert = 0.1
const = -9. /(4 * 3.14) * (1 - poisson) / lame
initial = np.array([Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd])

