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
from preparedata import preparation
from multiprocessing import Pool
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1" 
np.random.seed(1234)
Ncycmin = 20
Ncycmax = 60
nwalkers = 64
tiltErr = 1e-6
GPSErr = 1e-2
dxmin = 2.5
dxmax = 5
stations  = ['UWD']

if len(stations) > 1:
    pathtrunk = str(len(stations)) + 'st'
else:
    pathrunk = stations[0]

date = '06-16-2018'
date = '07-03-2018'
pathtrunk = pathrunk + '/' + date + '/'
pathrun = input('Name the path: ')
descr =input('Briefly describe the run: ')
pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
try:
    os.mkdir(pathgg)
    f =  open(pathgg + 'description.txt','w')
    f.write(descr)
    f.close()
except:
    print('Directory exists!')
if os.path.isfile(pathgg + 'progress.h5'):
    print('OhOh. Look like you started something...')
    sys.exit()


nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0 = preparation(pathgg,stations,date)
Nst = 1 + np.max(nstation)
dtx = np.zeros(len(tx))
dty = np.zeros(len(ty))
dtiltErr = 0

bndtiltconst = 1500
bndGPSconst = 100
bndtimeconst = 3600 * 24* 1
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

Nmax = 40
bounds = np.array([[+9,+11],[+8,+10],[+8,+10],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,10]]) #Vs,Vd,as,ad,taus,R5,R3,k_exp
ndim = len(bounds)
pos = np.zeros((nwalkers*10,ndim)) 
initial_rand = np.ones(ndim)
for i in range(len(bounds)):
    pos[:,i] = np.random.uniform(low = bounds[i][0],high = bounds[i][1],size = nwalkers * 10)
R1Initial = rhog * 10**pos[:,1] /(10**pos[:,2] * S)
lower =  rhog * (1 + R1Initial) * bounds[3,0] / (2 * R1Initial)
upper =  rhog * (1 + R1Initial) * bounds[3,1] / (2 * R1Initial)
pos[:,3] = np.random.uniform(low = lower,high = upper, size = nwalkers * 10)
ind = pos[:,3] < 5e+7
R1Initial = R1Initial[ind]
pos = pos[ind,:]
ind = np.random.uniform(len(pos),size = nwalkers)
ind = ind.astype(int)
pos = pos[ind,:]
R1Initial = R1Initial[ind]
lower = bounds[4,0] * 2 * R1Initial / (1 + R1Initial)
upper = bounds[4,1] * 2 * R1Initial / (1 + R1Initial)
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
pos = np.concatenate((offt,pos),axis = 1 )

nwalkers, ndim = pos.shape
filename = pathgg + "progress.h5"
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
    sampler.run_mcmc(pos, 150000, progress=True,thin = 10)
#    for sampler in sampler.sample(pos, 150000, progress=True,thin = 10):
#        if sampler.iteration % 100:
#            continue
#        print('Correlation time is ', sampler.get_autocorr_time())
#        print('And the ratio of the iteration to the corr. time is ', sampler.iteration / sampler.get_autocorr_time())

Obs = {}
Obs['tx'] =  tx
Obs['ty'] =  ty
Obs['GPS'] =  GPS

fix_par = np.array([x,y,ls,ld,pt,mu,rhog,const,S,Nmax,tiltErr,GPSErr])
stations  = ['UWD']
Obs['list_station'] = stations
Obs['fix_par'] = fix_par

with open('parameters.pickle','wb') as filen:
    pickle.dump(Obs,filen)
#
