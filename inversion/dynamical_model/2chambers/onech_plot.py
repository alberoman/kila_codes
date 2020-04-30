#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:35:08 2020

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
discardval = 1
thinval = 1

path_results = '../../../../results/onech/'
pathrun = 'alpha'
model_type = 'LF'

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

txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_onech(tTilt,tGPS,
                                              offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                                              VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp,
                                              x,y,
                                              ls,mu,
                                              rhog,const,S,nstation)
txmodlist = []
tymodlist = []
GPSmodlist = []
counter = 0
samples = reader.get_chain(thin = thinval,discard = discardval,flat = True)
for parameters in samples[np.random.randint(len(samples), size = 90)]:
    if Nst == 1:
        offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parameters
        offxSamp = np.array([offx1])
        offySamp = np.array([offy1])
    elif Nst == 2:
        offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSam,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parameters
        offxSamp = np.array([offx1],offx2)
        offySamp = np.array([offy1,offy2])
    elif Nst == 3:
        offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp = parameters
        offxSamp = np.array([offx1,offx2,offx3])
        offySamp = np.array([offy1,offy2,offy3])    

    txMod,tyMod,GPSMod = DirectModelEmcee_inv_onech(tTilt,tGPS,
                                              offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,
                                              VsExpSamp,ksExpSamp,R5ExpSamp,R3Samp,condsSamp,alphaSamp,
                                              x,y,
                                              ls,mu,
                                              rhog,const,S,nstation)
    counter = counter + 1   
    txmodlist.append(txMod)
    tymodlist.append(tyMod)
    GPSmodlist.append(GPSMod)
        
txspread = np.std(txmodlist,axis =0)
txmed = np.median(txmodlist,axis =0)
tyspread = np.std(tymodlist,axis =0)
tymed = np.median(tymodlist,axis =0)
GPSspread = np.std(GPSmodlist,axis =0)
GPSmed = np.median(GPSmodlist,axis =0)
for i in range(len(stations)):
    txStation = tx[nstation == i]
    tyStation = ty[nstation == i]
    txModbestStation= txModbest[ nstation == i]
    tyModbestStation= tyModbest[ nstation == i]
    txsprStation = txspread[nstation == i]
    txmedStation = txmed[nstation == i]
    tysprStation = tyspread[nstation == i]
    tymedStation = tymed[nstation == i]
    tTiltStation = tTilt[nstation == i]
    plt.figure()
    plt.plot(tTiltStation / (3600 * 24),txStation,'b')
    plt.plot(tTiltStation / (3600 * 24),txModbestStation,'r')
    plt.fill_between(tTiltStation /(3600 * 24),txmedStation-txsprStation,txmedStation + txsprStation,color='grey',alpha=0.5)
    plt.xlabel('Time [Days]')
    plt.ylabel('Tilt-x ' + stations[i])
    plt.savefig(pathfig + 'tx_' + stations[i] + '.pdf')
    plt.close('all')
    plt.figure()
    plt.plot(tTiltStation / (3600 * 24),tyStation,'b')
    plt.plot(tTiltStation / (3600 * 24),tyModbestStation,'r')
    plt.fill_between(tTiltStation / (3600 * 24),tymedStation - tysprStation,tymedStation + tysprStation,color='grey',alpha=0.5)
    plt.xlabel('Time [Days]')
    plt.ylabel('Tilt-y ' + stations[i])
    plt.savefig(pathfig + 'ty_' + stations[i] + '.pdf')
    plt.close('all')

plt.figure()
plt.plot(tGPS /(3600 * 24),GPS,'b')
plt.plot(tGPS / (3600 * 24),GPSModbest,'r')
plt.fill_between(tGPS/ (3600 * 24),GPSmed - GPSspread,GPSmed + GPSspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Piston displacement [m]')
plt.savefig(pathfig + 'GPS.pdf')
samples = reader.get_chain(thin = thinval,discard = discardval)
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)
plt.savefig(pathfig + 'chains.pdf')
plt.close('all')
samples = reader.get_chain(thin = thinval,discard = discardval,flat = True)
plt.figure()
#corner.corner(samples,truths = parmax)
#plt.savefig(pathfig + 'hist.pdf')
plt.close('all')

os.remove(pathgg + 'progress_temp.h5')

