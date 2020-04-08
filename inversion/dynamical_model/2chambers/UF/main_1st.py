#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
"""
def parameters_init():
    #Initialize parameters
    bndtiltconst = 1500
    bndGPSconst = 200
    bndtimeconst = 3600 * 24* 6
    bounds = np.array([[+8,+11],[+8,+11],[+8,+10],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,10]]) #Vs,Vd,as,ad,taus,R5,R3,k_exp

    rho = 2600
    g = 9.8
    rhog = rho * g
    #Conduit parameters
    ls = 4e+4
    ld = 2.5e+3
    mu = 100
    #Pressures and friction
    poisson = 0.25
    lame = 1e+9
    #Cylinder parameters
    Rcyl = 1.0e+3
    S = 3.14 * Rcyl**2 
    const = -9. /(4 * 3.14) * (1 - poisson) / lame
    ndim = len(bounds)
    return ls,ld,mu,rhog,const,S,bndtiltconst,bndGPSconst,bndtimeconst,bounds,ndim

def walkers_init(nwalkers,ndim):
    pos = np.zeros((nwalkers*10,ndim)) 
    initial_rand = np.ones(ndim)
    for i in range(len(bounds)):
        pos[:,i] = np.random.uniform(low = bounds[i][0],high = bounds[i][1],size = nwalkers * 10)
    R1Initial = rhog * 10**pos[:,0] /(10**pos[:,2] * S)
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
    return pos,nwalkers,ndim

import matplotlib.pyplot as plt
import numpy as np
from main_lib_UF import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from preparedata_UF import preparation_UF
from multiprocessing import Pool
import os
import sys

pathrun = '35_cycle_move_20'
os.environ["OMP_NUM_THREADS"] = "1" 
stations  = ['SDH']
date = '06-16-2018'
date = '07-03-2018'
thin = 10
np.random.seed(1234)
Ncycmin = 20
Ncycmax = 60
nwalkers = 64
tiltErr = 1e-6
GPSErr = 1e-2
dxmin = 3.0
dxmax = 4.0
Nmax = 35


if len(stations) > 1:
    pathtrunk = str(len(stations)) + 'st'
else:
    pathrunk = stations[0]


pathtrunk = pathrunk + '/' + date + '/'

pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
try:
    os.mkdir(pathgg)
    descr =input('Briefly describe the run: ')
    f =  open(pathgg + 'description.txt','w')
    f.write(descr)
    f.close()
except:
    print('Directory exists!')
if  os.path.isfile(pathgg + 'progress.h5'):
    nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0,bounds,bndtiltconst,bndGPSconst,bndtimeconst,tiltErr,GPSErr,fix_par,niter = pickle.load(open(pathgg + 'data.pickle','rb'))
    x,y,ls,ld,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par
    reader = emcee.backends.HDFBackend(pathgg + 'progress.h5', read_only=True)
    niter  = reader.iteration
    answer = input('Found past run with ' + str(niter) +' samples. Do you want to continue this run? Enter the number of iteration that additionally you want to do: ')
    moreiter = int(answer)
    print('Restarting!!!!')
    nwalkers,ndim = reader.shape
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                       args=(x,y,
                                        ls,ld,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,locTruth,locErr,nstation), backend = backend, pool = pool)
        sampler.run_mcmc(None, moreiter, progress=True,thin = thin)



else:
    niter = input('How many iteration you want to run? ')
    Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0 = preparation_UF(stations,date)
    ls,ld,mu,rhog,const,S,bndtiltconst,bndGPSconst,bndtimeconst,bounds,ndim = parameters_init()
    a_parameter = 20
    pos,nwalkers,ndim = walkers_init(nwalkers,ndim)
    fix_par = np.array([x,y,ls,ld,mu,rhog,const,S,Nmax,tiltErr,GPSErr])
    pickle.dump((nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0,bounds,bndtiltconst,bndGPSconst,bndtimeconst,tiltErr,GPSErr,fix_par,niter,a_parameter),open(pathgg + 'data.pickle','wb'))
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    
    move = emcee.moves.StretchMove(a=a_parameter)
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                        args=(x,y,
                                            ls,ld,mu,
                                            rhog,const,S,
                                            tTilt,tGPS,tx,ty,GPS,
                                            tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,locTruth,locErr,nstation),moves = [move], backend = backend, pool = pool)
       
        print('Running main sampling')
        sampler.run_mcmc(pos,niter, progress = True,thin = thin)
        