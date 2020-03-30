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
    #offtimeSamp,offGPSSamp,
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

def log_prior(param,S,rhog,bounds,bndtimeconst,bndGPSconst,locTr,locEr,Nobs):
   #offtimeSamp,offGPSSamp,
   offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
   kSamp = 10**kExpSamp
   R5Samp = 10**R5ExpSamp
   VsSamp= 10**VsExpSamp
   VdSamp = 10**VdExpSamp
   R1Samp = rhog * VdSamp /(kSamp*S)
   offs = np.array([offx1,offy1,])
   
   if bounds[0,0] < VsExpSamp < bounds[0,1] and bounds[1,0] < VdExpSamp < bounds[1,1] and bounds[2,0] < kExpSamp < bounds[2,1] and bounds[3,0] < R5ExpSamp < bounds[3,1] and 2* R1Samp* (Nobs -5) * (1 - R5Samp) / (R1Samp +1) +1   < R3Samp < 2* R1Samp* (Nobs + 30) * (1 - R5Samp) / (R1Samp +1) +1 and bounds[5,0] < condsSamp < bounds[5,1] and bounds[6,0] < conddSamp < bounds[6,1] and all(np.abs(offs)<3e+3):# and 0 < offtimeSamp < bndtimeconst and -bndGPSconst < offGPSSamp < 0 :                             
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
np.random.seed(1234)
bndtiltconst = 3000
bndGPSconst = 20
bndtimeconst = 3600 * 24* 20
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 4e+4
ld = 2.5e+3
mu = 500
#Pressures and friction
pt = 2.0e+7
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 1.0e+3
S = 3.14 * Rcyl**2 
const = -9. /(4 * 3.14) * (1 - poisson) / lame
tiltErr = 1e-5
GPSErr = 1e-2

nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS_SDH.pickle','rb'))
Nst = 1 + np.max(nstation)
tx = - tx
ty =  -ty
GPS = -GPS
dtx = np.zeros(len(tx))
dty = np.zeros(len(ty))
dtiltErr = 0

Nmax = 40
bounds = np.array([[+9,+11],[+7,+10],[+8,+11],[-2,0],[1.1,10],[1,10],[1,10]]) #Vs,Vd,as,ad,taus,R5,R3,k_exp
nwalkers = 64
ndim = len(bounds)
pos = np.zeros((nwalkers,ndim)) 
initial_rand = np.ones(ndim)
for i in range(len(bounds)):
    pos[:,i] = np.random.uniform(low = bounds[i][0],high = bounds[i][1],size = nwalkers)
R1Initial = rhog * 10**pos[:,1] /(10**pos[:,2] * S)
R5Initial = 10**pos[:,3]
lower =  2* R1Initial* (Nmax -3) * (1 - R5Initial) / (R1Initial +1) + 1
upper =  2* R1Initial* (Nmax +3) * (1 - R5Initial) / (R1Initial +1) + 1
pos[:,4] = np.random.uniform(low = lower,high = upper, size = nwalkers)
locs = np.zeros((nwalkers,6))
for i in range(len(locTruth)):
    locs[:,i] = np.random.normal(loc = locTruth[i], scale =locErr[i],size = nwalkers)
pos = np.concatenate((locs,pos),axis = 1)

offtilt = np.zeros((nwalkers, Nst * 2))

for i in range(Nst * 2):
    offtilt[:,i] = np.random.uniform(low = -bndtiltconst, high = bndtiltconst, size =nwalkers)
pos = np.concatenate((offtilt,pos),axis = 1)

offsGPS = np.random.uniform(low = -bndGPSconst , high = 0,size = (nwalkers,1))
#pos = np.concatenate((offsGPS,pos),axis = 1 )

offt = np.random.uniform(low = 0,high = bndtimeconst,size = (nwalkers,1))
#pos = np.concatenate((offt,pos),axis = 1 )

nwalkers, ndim = pos.shape
filename = "progress_SDH.h5"
backend = emcee.backends.HDFBackend(filename)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x,y,
                                        ls,ld,pt,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation), backend = backend, pool = pool)
    #print('Running burn-in')
    #pos, _, _ = sampler.run_mcmc(pos, 2500,progress=True)
    #sampler.reset()
    print('Running main sampling')
    sampler.run_mcmc(pos, 300000, progress=True)
samples = sampler.get_chain(thin =16)
samplesFlat = sampler.get_chain(flat = True,thin = 16)
Obs = {}
Obs['tx'] =  tx
Obs['ty'] =  ty
Obs['GPS'] =  GPS
fix_par = np.array([x,y,ls,ld,pt,mu,rhog,const,S])

with open('results_SDH.pickle','wb') as filen:
    pickle.dump((tTilt,tGPS,Obs,samples,samplesFlat,fix_par,Nmax),filen)
#
