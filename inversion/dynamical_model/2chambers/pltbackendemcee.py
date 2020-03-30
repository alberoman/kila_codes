#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:36:49 2020

@author: aroman
"""
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import pickle
from main_lib import DirectModelEmcee_inv
def plotter(samples,pathfigs):
    nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS.pickle','rb'))
    tx = -tx
    ty = -ty
    GPS = -GPS
    
    rho = 2600
    g = 9.8
    rhog = rho * g
    #Conduit parameters
    ls = 4e+4
    ld = 2.5e+3
    mu = 500
    #Pressures and friction
    pt = 2.0e+7
    poisson = 0.25
    lame = 1e+9
    #Cylinder parameters
    Rcyl = 1.0e+7
    S = 3.14 * Rcyl**2 
    const = -9. /(4 * 3.14) * (1 - poisson) / lame
    
    plt.figure()
    for parameters in samples[np.random.randint(np.shape(samplesFlat)[1], size = 30)]:
        offtimeSamp,offGPSSamp,offx1,offx2,offx3,offy1,offy2,offy3,xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd = parameters
        offx = np.array([offx1,offx3,offx3])
        offy = np.array([offy1,offy2,offy3])
        txMod,tyMod,GPSMod = DirectModelEmcee_inv(tTilt,tGPS,
                        offtimeSamp,offGPSSamp,
                        offx,offy,xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
                        rhog,const,S,nstation)
        plt.plot(tTilt,txMod[::],color = 'k', alpha=0.2)
    plt.plot(tTilt,txMod[::])
    plt.savefig(pathfigs +'model.png')

pathfigs = 'figs_UWD/' 
filename = "progress_SDH.h5"
reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain(thin =10)
samplesFlat = reader.get_chain(flat = True,thin = 10)
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
corner.corner(samplesFlat)
plt.savefig(pathfigs +'hist.png')
plt.close('all')

#plotter(samplesFlat,pathfigs)

