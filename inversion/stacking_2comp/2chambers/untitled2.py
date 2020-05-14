#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:25:19 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:23:01 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:36:49 2020

@author: aroman
"""
import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from disconnected_lib import *
import os
import corner
from shutil import copyfile
discardval = 1
thinval = 1

path_results = '../../../../results/disc/'
pathrun = 'test'
model_type = 'UF'

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

Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
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
samples = reader.get_chain(flat = True,discard = discardval)
parmax = samples[np.argmax(reader.get_log_prob(flat = True,discard = discardval))]
if Nst == 1:
    deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
    offxSamp = np.array([offx1])
    offySamp = np.array([offy1])
elif Nst == 2:
    deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
    offxSamp = np.array([offx1],offx2)
    offySamp = np.array([offy1,offy2])
elif Nst == 3:
    deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
    offxSamp = np.array([offx1,offx2,offx3])
    offySamp = np.array([offy1,offy2,offy3])    
if model_type == 'UF':
    txModbest,tyModbest,GPSModbest,ps,pd = DirectModelEmcee_inv_disc_diagno(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
elif model_type == 'LF':
    txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_disc(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
txmodlist = []
tymodlist = []
GPSmodlist = []
counter = 0
samples = reader.get_chain(thin = thinval,discard = discardval,flat = True)
for parameters in samples[np.random.randint(len(samples), size = 90)]:
    if Nst == 1:
        deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
        offxSamp = np.array([offx1])
        offySamp = np.array([offy1])
    elif Nst == 2:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
        offxSamp = np.array([offx1],offx2)
        offySamp = np.array([offy1,offy2])
    elif Nst == 3:
        deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
        offxSamp = np.array([offx1,offx2,offx3])
        offySamp = np.array([offy1,offy2,offy3])    
    if model_type == 'UF':
        txMod,tyMod,GPSMod,ps,pd = DirectModelEmcee_inv_disc_diagno(tTilt,tGPS,
                        deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                        VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                        x,y,
                        ls,ld,mu,
                        rhog,const,S,nstation)
        
