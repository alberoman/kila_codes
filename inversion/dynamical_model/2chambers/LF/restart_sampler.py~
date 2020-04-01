#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:07:05 2020

@author: aroman
"""
import numpy as np
import emcee
import pickle
from multiprocessing import Pool
from main_lib import *

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
nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS_UWD.pickle','rb'))

Nst = 1 + np.max(nstation)
tx = - tx
ty =  -ty
GPS = -GPS
dtx = np.zeros(len(tx))
dty = np.zeros(len(ty))
dtiltErr = 0

Nmax = 40
bounds = np.array([[+9,+11],[+7,+10],[+8,+11],[-2,0],[1.1,10],[1,10],[1,10]]) 
filename = "progress_UWD_constr.h5"
backend = emcee.backends.HDFBackend(filename)
nwalkers,ndim = backend.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x,y,
                                        ls,ld,pt,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation), backend = backend, pool = pool)
    sampler.run_mcmc(None, 500000,progress=True)
    
samples = sampler.get_chain(thin =30)
samplesFlat = sampler.get_chain(flat = True,thin = 30)
Obs = {}
Obs['tx'] =  tx
Obs['ty'] =  ty
Obs['GPS'] =  GPS
fix_par = np.array([x,y,ls,ld,pt,mu,rhog,const,S])

with open('results_UWD_constr.pickle','wb') as filen:
    pickle.dump((tTilt,tGPS,Obs,samples,samplesFlat,fix_par,Nmax),filen)


