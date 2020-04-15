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
from main_lib import *
import os
import corner
from shutil import copyfile
discardval = 1
thinval = 1

path_results = '../../../../results/'
pathrun = 'press'
model_type = 'LF'

stations  = ['UWD']
date = '07-03-2018'
flaglocation = 'F'      # This is the flag for locations priors, F for Uniform, N for Normal

if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]



pathtrunk = pathrunk + '/' + date + '/'

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
bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = pickle.load(open(pathgg + 'parameters.pickle','rb'))
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
samples = reader.get_chain(flat = True)
parmax = samples[np.argmax(reader.get_log_prob(flat = True))]
deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax

offxSamp = np.array([offx1])
offySamp = np.array([offy1])
if model_type == 'UF':
    txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_UF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
elif model_type == 'LF':
    txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)

txmodlist = []
tymodlist = []
GPSmodlist = []
samples = reader.get_chain(thin = thinval,discard = discardval,flat = True)
for parameters in samples[np.random.randint(len(samples), size = 100)]:
    
    deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parameters

    offx = np.array([offx1,])
    offy = np.array([offy1,])
    if model_type == 'UF':
        txMod,tyMod,GPSMod = DirectModelEmcee_inv_UF(tTilt,tGPS,
                        deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                        VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                        x,y,
                        ls,ld,mu,
                        rhog,const,S,nstation)
    elif model_type == 'LF':
        txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                        deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                        VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
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
plt.plot(tTilt / (3600 * 24),txModbest,'r')
plt.fill_between(tTilt /(3600 * 24),txmed-txspread,txmed + txspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')




plt.ylabel('Tilt-x' )

plt.savefig(pathfig + 'tx.pdf')
plt.figure(2)
plt.plot(tTilt / (3600 * 24),ty,'b')
plt.plot(tTilt / (3600 * 24),tyModbest,'r')
plt.fill_between(tTilt / (3600 * 24),tymed - tyspread,tymed + tyspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Tilt-y')
plt.savefig(pathfig + 'ty.pdf')

plt.figure(3)
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
corner.corner(samples,truths = parmax)
plt.savefig(pathfig + 'hist.pdf')
plt.close('all')

os.remove(pathgg + 'progress_temp.h5')

