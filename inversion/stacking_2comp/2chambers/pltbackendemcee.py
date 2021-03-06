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
from main_lib_UF import *
import os
import corner
from shutil import copyfile
discardval = 10000
thinval = 1
pathgg = 'UWD/07-03-2018//45_cycle_locErr_5_sigma_10_notimeoffset_noGPS/'
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

list_pickle = pickle.load(open(pathgg + 'data.pickle','rb'))

if len(list_pickle) == 19:
    nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0,bounds,bndtiltconst,bndGPSconst,bndtimeconst,tiltErr,GPSErr,fix_par,niter = list_pickle
elif len(list_pickle) == 20: 
    nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0,bounds,bndtiltconst,bndGPSconst,bndtimeconst,tiltErr,GPSErr,fix_par,niter,a_parameter = list_pickle

x,y,ls,ld,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par

dtiltErr = 0
Nmax = 40
filename = 'progress_temp.h5'
#Getting sampler info
#nwalkers,ndim = reader.shape
#backend = emcee.backends.HDFBackend(pathgg + filename)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
#                                   args=(x,y,
#                                    ls,ld,mu,
#                                    rhog,const,S,
#                                    tTilt,tGPS,tx,ty,GPS,
#                                    tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,locTruth,locErr,nstation), backend = backend)

reader = emcee.backends.HDFBackend(pathgg + filename, read_only = True )
nwalkers,ndim = reader.shape
samples = reader.get_chain(flat = True)
parmax = samples[np.argmax(reader.get_log_prob(flat = True))]
if len(parmax)==16:
    offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax
    offtime = 0
elif len(parmax)==17:
    offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax

offxSamp = np.array([offx1])
offySamp = np.array([offy1])
txMod,tyMod,GPSMod = DirectModelEmcee_inv_UF(tTilt,tGPS,
                                              offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)

txmodlist = []
tymodlist = []
GPSmodlist = []
for parameters in samples[np.random.randint(len(samples), size = 100)]:
    if len(parmax)==16:
        offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
        offtime = 0
    elif len(parmax)==17:
        offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters
    offx = np.array([offx1,])
    offy = np.array([offy1,])
    txMod,tyMod,GPSMod = DirectModelEmcee_inv_UF(tTilt,tGPS,
                        offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                        VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                        x,y,
                        ls,ld,mu,
                        rhog,const,S,nstation)
        
    txmodlist.append(txMod)
    tymodlist.append(tyMod)
    GPSmodlist.append(GPSMod)
        
txspread = np.std(txmodlist,axis =0)
txmed = np.median(txmodlist,axis =0)
tyspread = np.std(tymodlist,axis =0)
tymed = np.median(tymodlist,axis =0)
GPSspread = np.std(GPSmodlist,axis =0)
GPSmed = np.median(GPSmodlist,axis =0)

plt.figure(1)
plt.plot(tTilt / (3600 * 24),tx,'b')
plt.plot(tTilt / (3600 * 24),txMod,'r')
plt.fill_between(tTilt /(3600 * 24),txmed-txspread,txmed + txspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Tilt-x' )

plt.savefig(pathfig + 'tx.pdf')
plt.figure(2)
plt.plot(tTilt / (3600 * 24),ty,'b')
plt.plot(tTilt / (3600 * 24),tyMod,'r')
plt.fill_between(tTilt / (3600 * 24),tymed - tyspread,tymed + tyspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Tilt-y')
plt.savefig(pathfig + 'ty.pdf')

plt.figure(3)
plt.plot(tGPS /(3600 * 24),GPS,'b')
plt.plot(tGPS / (3600 * 24),GPSMod,'r')
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
#corner.corner(samples)
#plt.savefig(pathfig + 'hist.pdf')
plt.close('all')

os.remove(pathgg + 'progress_temp.h5')

