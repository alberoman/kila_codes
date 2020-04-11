#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:30:30 2020

@author: aroman
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from main_lib_UF import *
import os
import corner
from shutil import copyfile
discardval = 1
thinval = 1
pathgg = 'UWD/07-03-2018/45_cycle_locErr_5_sigma_10_notimeoffset_noG/'
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
bndtiltconst = 1500
bndGPSconst = 100
bndtimeconst = 3600 * 24* 1

nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0,bounds,bndtiltconst,bndGPSconst,bndtimeconst,tiltErr,GPSErr,fix_par,niter = pickle.load(open(pathgg + 'data.pickle','rb'))
x,y,ls,ld,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par
#tTilt = np.linspace(tTilt[0],tTilt[-1],len(tTilt)*50)
dtiltErr = 0
Nmax = 40
filename = 'progress_temp.h5'
reader = emcee.backends.HDFBackend(pathgg + filename, read_only = True )
nwalkers,ndim = reader.shape
samples = reader.get_chain(flat = True)
parmax = samples[np.argmax(reader.get_log_prob(flat = True))]
if len(parmax) ==17:
    offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
elif len(parmax) ==16:
    offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
offxSamp = np.array([offx1])
offySamp = np.array([offy1])
txMod,tyMod,gpsMod,ps,pd,coeffxs,coeffys,coeffxd,coeffyd = DirectModelEmcee_inv_UF_debug(tTilt,tGPS,
                                              offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
ind = np.argsort(reader.get_log_prob(flat = True))
models = samples[ind]
models = models[::-1]
models = models[:300]

plt.plot(tTilt,tx)
for par in models:
    if len(parmax) ==17:
        offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp =par
    elif len(parmax) ==16:
        offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = par
        
    txMod,tyMod,gpsMod,ps,pd,coeffxs,coeffys,coeffxd,coeffyd = DirectModelEmcee_inv_UF_debug(tTilt,tGPS,
                                              offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
    plt.plot(tTilt,txMod)
    