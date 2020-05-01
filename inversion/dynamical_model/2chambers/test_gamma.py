#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:35:58 2020

@author: aroman
"""

import matplotlib.pyplot as plt
import numpy as np
from gamma_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from preparedata import preparation
from multiprocessing import Pool
import os
import sys
np.random.seed(1234567284)
path_results = '../../../../results/gamma/'

    

pathrun = 'gamma'

model_type = 'LF'

stations  = ['UWD','SDH','IKI']
date = '07-03-2018'
flaglocation = 'N'      # This is the flag for locations priors, F for Uniform, N for Normal

def parameters_init(mt):
    #Initialize parameters
    dxmax = 5.0
    dxmin = 3.0

    Ncycmin = 20
    Ncycmax = 60

    if mt == 'UF':
        bounds = np.array([[+8,10],[+9,+12],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20],[0.5,2.0],[0,1]])
        boundsLoc = np.array([[-1000,1000],[1200,2500],[500,1500],[-1500,+1500],[-1500,2500],[2000,4000]])
    elif mt == 'LF':                                                 
        bounds = np.array([[+8,+12],[+8,+10],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20],[0.5,2.0],[0,1]])
        boundsLoc = np.array([[-1000,+1000],[-1000,1000],[2000,5000],[-500,0],[1500,2000],[500,1300]])

    bndtiltconst = 10000
    bndGPSconst = 100
    bnddeltaP0 = 1e+7
    tiltErr = 1e-5
    GPSErr = 1e-2
    locErrFact = 5
    
    a_parameter = 2
    thin = 10
    nwalkers = 64

    rho = 2600
    g = 9.8
    #Conduit parameters
    ls = 4e+4
    ld = 2.5e+3
    mu = 100
    #Elastic properties
    poisson = 0.25
    lame = 1e+9
    #Cylinder parameters
    Rcyl = 1.0e+3
    ndim = len(bounds)
    const = -9. /(4 * 3.14) * (1 - poisson) / lame
    S = 3.14 * Rcyl**2 
    rhog = rho * g
    return bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bnddeltaP0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog


bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = parameters_init(model_type)
VsExpSamp = 10
VdExpSamp = 9
ksExpSamp = 9
kdExpSamp = 9
pspdSamp = 2e+6
R3Samp = 3
condsSamp = 5
conddSamp = 3
tTilt = np.linspace(0,3600 * 24 *60, 600)
tGPS = np.linspace(0,3600 * 24 *60, 60)
gammaSamp = 0.2
alphaSamp = 1
VdSamp = 10**VdExpSamp
VsSamp = 10**VsExpSamp
kdsamp = 10**kdExpSamp
ksSamp = 10**ksExpSamp
R1Samp = rhog * VdSamp / (kdsamp)
ps,pd,GPSMod = DirectModelEmcee_inv_LF_diagno(tTilt,tGPS,
                                              deltap0Samp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,pspdSamp,R3Samp,condsSamp,conddSamp,alphaSamp,gammaSamp,
                                              ls,ld,mu,
                                              rhog,const,S) 
plt.plot(tTilt,ps,'b')
plt.plot(tTilt,pd,'r')