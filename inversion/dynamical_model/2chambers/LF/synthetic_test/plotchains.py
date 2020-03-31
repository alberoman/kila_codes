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
from synthetic_test_lib import *
rho = 2600
g = 9.8
rhog = rho * g
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
def plotter(samples,tTilt,tGPS,Obs,truth,fix_par,Nobs):
    plt.figure(1)
    plt.plot(Obs['tx'][:700])
    for parameters in samples[np.random.randint(len(samples), size=30)]:
        xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd = parameters
        #xs,ys,ds,xd,yd,dd,Vs_exp,Vd_exp,k_exp,R5_exp,R3,conds,condd = parameters
        xstation,ystation,ls,ld,pt,mu,rhog,const,S = fix_par
        txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                        xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        xstation,ystation,
                        ls,ld,pt,mu,
                        rhog,const,S)
        plt.plot(txMod[:700],color = 'k', alpha=0.3)
        

pathfigs = 'figs/' 
with open('samples.pickle','rb') as filen:
    tTilt,tGPS,Obs,samples,samplesFlat,truth,fix_par,Nobs = pickle.load(filen)

ndim =  np.shape(samples)[2]

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig(pathfigs + 'chains.png')
plt.close('all')
corner.corner(samplesFlat, truths= truth)
plt.savefig(pathfigs +'hist.png')
plt.close('all')
tTilt = np.concatenate(tTilt)
plotter(samplesFlat,tTilt,tGPS,Obs,truth,fix_par,Nobs)

