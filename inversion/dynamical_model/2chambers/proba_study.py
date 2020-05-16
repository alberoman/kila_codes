#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:34:52 2020

@author: aroman
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from connected_lib import *
import os
from scipy import signal
import corner
from shutil import copyfile
discardval = 1
thinval = 1

path_results = '../../../../results/conn/'
pathrun = 'gpsscale'
model_type = 'LF'
limit = 10
stations  = ['UWD','SDH','IKI']
date = '07-03-2018'
flaglocation = 'N'      # This is the flag for locations priors, F for Uniform, N for Normal

if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]



pathtrunk = pathtrunk + '/' + date + '/'

pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
pathgg = path_results + pathgg

copyfile(pathgg + 'progress.h5', pathgg + 'progress_temp.h5' )
pathfig = pathgg + 'figs/'

try:
    os.mkdir(pathfig)
except:
    print('Directory already exist')
#parameters = pickle.load(open(pathgg + 'parameters.pickle','rb')) 
#fix_par = parameters['fix_par']
##tx = parameters['tx']
#ty = parameters['ty']
#dtx = np.zeros(np.shape(tx))
#dty = np.zeros(np.shape(ty))

#GPS = parameters['GPS']
#x,y,ls,ld,pt,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par

np.random.seed(1234)
b, a = signal.butter(2, 0.3)

Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
#tx = signal.filtfilt(b, a,ty)
#ty = signal.filtfilt(b, a,tx)
a = pickle.load(open(pathgg + 'parameters.pickle','rb'))
if len(a)== 17:
    bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
    boundsLoc = 0
elif len(a)== 18:
    bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
filename = 'progress_temp.h5'
#Getting sampler info
#nwalkers,ndim = reader.shape
#backend = emcee.backends.HDFBackend(pathgg + filename)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
#                                   args=(x,y,
#                                            ls,ld,mu,
#                                            rhog,const,S,
#                                            tTilt,tGPS,tx,ty,GPS,
#                                            tiltErr,GPSErr,bounds,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation), backend = backend)

reader = emcee.backends.HDFBackend(pathgg + filename, read_only = True )
nwalkers,ndim = reader.shape
samples = reader.get_chain()
proba = reader.get_log_prob()
samples = samples[:20000,:,:]
proba = proba[:20000,:]
ind = np.unravel_index(np.argmax(proba),proba.shape)
parmax = samples[ind[0],ind[1],:]
txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,gpsscaleSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
plt.plot(tTilt,tx)
plt.plot(tTilt,txModbest)
