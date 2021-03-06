#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:12:23 2020

@author: aroman
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from onech_lib import *
import os
import corner
from shutil import copyfile
discardval = 10000
thinval = 1

path_results = '../../../../results/onech/'
pathrun = 'sigma'
model_type = 'UF'

stations  = ['UWD']
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

Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
a = pickle.load(open(pathgg + 'parameters.pickle','rb'))

bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,locErrFact,a_parameter,thin,nwalkers,ls,mu,ndim,const,S,rhog = a
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
samples = reader.get_chain(flat = True,discard = discardval)
parmax = samples[np.argmax(reader.get_log_prob(flat = True,discard = discardval))]
if Nst == 1:
    offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parmax
    offxSamp = np.array([offx1])
    offySamp = np.array([offy1])
elif Nst == 2:
    offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSam,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parmax
    offxSamp = np.array([offx1],offx2)
    offySamp = np.array([offy1,offy2])
elif Nst == 3:
    offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parmax
    offxSamp = np.array([offx1,offx2,offx3])
    offySamp = np.array([offy1,offy2,offy3])    
#tTilt = np.linspace(0,np.max(tTilt)*4,len(tTilt)*4)
#x = x[0] * np.ones(len(tTilt))
#y = y[0] * np.ones(len(tTilt))
#nstation = nstation[0]*np.ones(len(tTilt))
ps = DirectModelEmcee_inv_diagno(tTilt,tGPS,
                                              offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                                              VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp,
                                              x,y,
                                              ls,mu,
                                              rhog,const,S,nstation)
plt.plot(tTilt,ps)