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


np.random.seed(1234)
bndtiltconst = 3000
bndGPSconst = 80
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

nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS_UWD.pickle','rb'))
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
pos = np.concatenate((offsGPS,pos),axis = 1 )

offt = np.random.uniform(low = 0,high = bndtimeconst,size = (nwalkers,1))
#pos = np.concatenate((offt,pos),axis = 1 )

nwalkers, ndim = pos.shape
filename = "progress_UWD_offGPS.h5"
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
    sampler.run_mcmc(pos, 10000, progress=True)
samples = sampler.get_chain(thin =2)
samplesFlat = sampler.get_chain(flat = True,thin = 2)
Obs = {}
Obs['tx'] =  tx
Obs['ty'] =  ty
Obs['GPS'] =  GPS
fix_par = np.array([x,y,ls,ld,pt,mu,rhog,const,S])

with open('results_UWD_offGPS.pickle','wb') as filen:
    pickle.dump((tTilt,tGPS,Obs,samples,samplesFlat,fix_par,Nmax),filen)
#
