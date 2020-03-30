#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:14:57 2020

@author: aroman
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import corner
from main_lib import DirectModelEmcee_inv
rho = 2600
g = 9.8
rhog = rho * g
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
def plotter(samples,tTilt,tGPS,Obs,fix_par,Nobs,pathfigs):
    nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS_UWD.pickle','rb'))
    tx = -tx
    ty = -ty
    GPS = -GPS
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    for parameters in samples[np.random.randint(len(samples), size = 20)]:
        #offtimeSamp,offGPSSamp,
        offGPS,offx1,offy1,xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd = parameters
        offx = np.array([offx1,])
        offy = np.array([offy1,])
        x,y,ls,ld,pt,mu,rhog,const,S = fix_par
        txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                        offGPS,offx,offy,xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
                        rhog,const,S,nstation)
        plt.figure(1)
        plt.plot(tTilt,txMod,color = 'k', alpha=0.2)
        
        plt.figure(2)
        plt.plot(tTilt,tyMod,color = 'k', alpha=0.2)
        plt.figure(3)
        plt.plot(tGPS,GPSMod,color = 'k', alpha=0.2)
    plt.figure(1)
    plt.plot(tTilt,tx)
    plt.savefig(pathfigs +'tx.png')
    plt.figure(2)
    plt.plot(tTilt,ty)
    plt.savefig(pathfigs +'ty.png')
    plt.figure(3)
    plt.plot(tGPS,GPS)
    plt.savefig(pathfigs +'GPS.png')

pathfigs = 'figs_UWD_offGPS/' 
with open('results_UWD_offGPS.pickle','rb') as filen:
    tTilt,tGPS,Obs,samples,samplesFlat,fix_par,Nobs = pickle.load(filen)

ndim =  np.shape(samples)[2]
#
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)


axes[-1].set_xlabel("step number")
plt.savefig(pathfigs + 'chains.png')
plt.close('all')
corner.corner(samplesFlat)
plt.savefig(pathfigs +'hist.png')
plt.close('all')
plotter(samplesFlat,tTilt,tGPS,Obs,fix_par,Nobs,pathfigs)

