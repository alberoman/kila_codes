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
from main_lib import *
import os
import corner
discardval = 1000
thinval = 1
pathgg = 'UWD/07-03-2018/locErr7.5/'
pathfig = pathgg + 'figs/'
try:
    os.mkdir(pathfig)
except:
    print('Directory already exist')
parameters = pickle.load(open(pathgg + 'parameters.pickle','rb')) 
fix_par = parameters['fix_par']
tx = parameters['tx']
ty = parameters['ty']
dtx = np.zeros(np.shape(tx))
dty = np.zeros(np.shape(ty))

GPS = parameters['GPS']
x,y,ls,ld,pt,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par

np.random.seed(1234)
bndtiltconst = 1500
bndGPSconst = 100
bndtimeconst = 3600 * 24* 1

nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))


dtiltErr = 0
Nmax = 40
filename = 'progress.h5'
bounds = np.array([[+9,+11],[+7,+10],[+8,+11],[-2,0],[1.1,10],[1,10],[1,20]]) 
backend = emcee.backends.HDFBackend(pathgg + filename)
nwalkers,ndim = backend.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x,y,
                                        ls,ld,pt,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation), backend = backend)

samples = sampler.get_chain(flat = True)
parmax = samples[np.argmax(sampler.flatlnprobability)],
offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax[0]
offxSamp = np.array([offx1])
offySamp = np.array([offy1])
txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,pt,mu,
                                              rhog,const,S,nstation)

txmodlist = []
tymodlist = []
GPSmodlist = []
for parameters in samples[np.random.randint(len(samples), size = 100)]:
        #offtimeSamp,offGPSSamp,
        offtime,offGPS,offx1,offy1,xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd = parameters
        offx = np.array([offx1,])
        offy = np.array([offy1,])
        txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                        offtime,offGPS,offx,offy,xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
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

samples = sampler.get_chain(thin = thinval,discard = discardval,flat = True)
plt.figure()
corner.corner(samples)
plt.savefig(pathfig + 'hist.pdf')

samples = sampler.get_chain(thin = thinval,discard = discardval)
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)
plt.savefig(pathfig + 'chains2.pdf')


