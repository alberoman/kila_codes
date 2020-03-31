#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

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

os.environ["OMP_NUM_THREADS"] = "1" 

def log_likelihood(param,
                   xstation,ystation,
                   ls,ld,pt,mu,
                   rhog,const,S,
                   tTilt,tGPS,txObs,tyObs,GPSObs,
                   tiltErr,GPSErr):
    
    xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
    txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                                              xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              xstation,ystation,
                                              ls,ld,pt,mu,
                                              rhog,const,S)
    sigma2Tilt = tiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx = -0.5 * np.sum((txObs - txMod) ** 2 / sigma2Tilt,0)  -len(txObs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    likety = -0.5 * np.sum((tyObs - tyMod) ** 2 / sigma2Tilt,0)  -len(tyObs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
    return np.sum(liketx) + np.sum(likety) + likeGPS

def log_prior(param,S,rhog,bounds,locTr,locEr,Nobs):
   xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = param
   kSamp = 10**kExpSamp
   R5Samp = 10**R5ExpSamp
   VsSamp= 10**VsExpSamp
   VdSamp = 10**VdExpSamp
   R1Samp = rhog * VdSamp /(kSamp*S)

   if bounds[0,0] < VsExpSamp < bounds[0,1] and bounds[1,0] < VdExpSamp < bounds[1,1] and bounds[2,0] < kExpSamp < bounds[2,1] and bounds[3,0] < R5ExpSamp < bounds[3,1] and bounds[4,0]   < R3Samp < bounds[4,1] and bounds[5,0] < condsSamp < bounds[5,1] and bounds[6,0] < conddSamp < bounds[6,1]:
       logprob =   np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(xsSamp-locTr[0])**2/locEr**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(ysSamp-locTr[1])**2/locEr**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(dsSamp-locTr[2])**2/locEr**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(xdSamp-locTr[3])**2/locEr**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(ydSamp-locTr[4])**2/locEr**2
       logprob = logprob +  np.log(1.0/(np.sqrt(6.28)*locEr))-0.5*(ddSamp-locTr[5])**2/locEr**2
       return logprob
   return -np.inf

def log_probability(param,
                    xstation,ystation,
                    ls,ld,pt,mu,
                    rhog,const,S,
                    tTilt,tGPS,tx,ty,GPS,
                    tiltErr,GPSErr,locTruth,locErr,bounds,Nmax):
    
    lp = log_prior(param,S,rhog,bounds,locTruth,locErr,Nmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp 
  
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
locErr = 500
tiltErr = 1e-5
dtiltErr = 1e-8
GPSErr = 1e-2

xsh = xSource[0]
xde = xSource[1]
ysh = ySource[0]
yde = ySource[1]
dsh = depthSource[0]
dde = depthSource[1]

tTilt,tx1,ty1,tx2,ty2,GPS,tGPS,Nmax = Twochambers_syntethic_LF(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,conds,condd,Vs,Vd,ks,taus,taud,mu,pt,Rcyl,rho,g,poisson,lame,tiltPert)
print('Number of cycles is ', Nmax)
tx = np.concatenate((tx1,tx2),axis = 0)
ty = np.concatenate((ty1,ty2),axis = 0)
x = np.array([xstation1,xstation2])
y = np.array([ystation1,ystation2])

print('The length of the model is ', len(tTilt))
txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                        xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
                        rhog,const,S)
# dtx1 = np.diff(tx1) / np.diff(tTilt)
# dty1 = np.diff(ty1) / np.diff(tTilt)
# dtx2 = np.diff(tx2) / np.diff(tTilt)
# dty2 = np.diff(ty2) / np.diff(tTilt)
tx = np.concatenate((np.expand_dims(tx1,1),np.expand_dims(tx2,1)),axis = 1)
ty = np.concatenate((np.expand_dims(ty1,1),np.expand_dims(ty2,1)),axis = 1)

plt.plot(tTilt,tx,tTilt,txMod,'r')
initial = np.array([xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd])
#Initializing model parameters
bounds = np.array([[+9,+11],[+8,+10],[+8,+10],[-2,0],[1.1,5],[1,5],[1,5]]) #Vs,Vd,as,ad,taus,R5,R3,k_exp
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
#Initializing location sources
locTruth =np.array([xsh,ysh,dsh,xde,yde,dde])
locs = np.zeros((nwalkers,6))
for i in range(len(locTruth)):
    locs[:,i] = np.random.normal(loc = locTruth[i], scale =locErr,size = nwalkers)
pos = np.concatenate((locs,pos),axis = 1)



nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x,y,
                                        ls,ld,pt,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,locTruth,locErr,bounds,Nmax),pool = pool)
 #   print('Running burn-in')
 #   pos, _, _ = sampler.run_mcmc(pos, 100)
 #   sampler.reset()
    print('Running main sampling')
    sampler.run_mcmc(pos, 1000, progress=True)
samples = sampler.get_chain()
samplesFlat = sampler.get_chain(flat = True)
truth = initial
fix_par = np.array([x,y,ls,ld,pt,mu,rhog,const,S])
Obs = {}
Obs['tx'] =  tx
Obs['ty'] =  ty

# Obs['dtx1'] =  dtx1
# Obs['dty1'] =  dty1
# Obs['dtx2'] =  dtx2
# Obs['dty2'] =  dty2
Obs['GPS'] =  GPS
with open('samples.pickle','wb') as filen:
    pickle.dump((tTilt,tGPS,Obs,samples,samplesFlat,truth,fix_par,Nmax),filen)

sampler.get_autocorr_time()


