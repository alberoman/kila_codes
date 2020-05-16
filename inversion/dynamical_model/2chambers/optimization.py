#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:05:53 2020

@author: aroman
"""
from scipy import optimize
import numpy as np
from conn_lib import log_likelihood_LF
import pickle
from preparedata import preparation
import emcee
stations  = ['UWD','SDH','IKI']
date = '07-03-2018'
model_type = 'LF'
pathbestfit = '/Users/aroman/work/kilauea_2018/results/priorsN/LF/3st/07-03-2018/test/'
filename = 'progress.h5'

def parameters_init(mt):
    #Initialize parameters
    dxmax = 5.0
    dxmin = 3.0

    Ncycmin = 20
    Ncycmax = 60

    if mt == 'UF':
        bounds = np.array([[+8,10],[+8,+12],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[8,50]])
        boundsLoc = np.array([[-1000,1000],[1200,2500],[500,1500],[-1500,+1500],[-1500,2500],[2000,4000]])
    elif mt == 'LF':                                                 
        bounds = np.array([[+8,+12],[+8,+10],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[8,50]])
        boundsLoc = np.array([[-1000,+1000],[-1000,1000],[2000,5000],[-500,0],[1500,2000],[500,1300]])

    bndtiltconst = 2000
    bndGPSconst = 200
    bnddeltaP0 = 4e+7
    tiltErr = 1e-5
    GPSErr = 1e-2
    locErrFact = 5
    
    a_parameter = 5
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
Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0 = preparation(stations,date,locErrFact,model_type)

fix_par= x,y,ls,ld,mu,rhog,const,S,tTilt,tGPS,tx,ty,GPS,tiltErr,GPSErr,nstation

bnds = ((-bndp0,bndp0),(-100,100),(-2000,2000),(-2000,2000),(-2000,2000),(-2000,2000),(-2000,2000),(-2000,2000),(-1000,1000),(-3000,1000),(2000,5000),(-1000,1000),(1000,2000),(300,1500),(+9,+12),(9,12),(8,11),(1e+5,5e+6),(1e+6,3e+7),(1,10),(1,30),(0.2,0.7))
minimizer_kwargs = {"method":"L-BFGS-B", "args":fix_par,'bounds':bnds}
initial = np.array([0,-40,0,0,0,0,0,0,])
soln = minimize(-log_likelihood_LF, initial, args=(x, y, yerr))

#res = optimize.basinhopping(-log_likelihood_LF_opt,parmax, minimizer_kwargs=minimizer_kwargs,disp = True)