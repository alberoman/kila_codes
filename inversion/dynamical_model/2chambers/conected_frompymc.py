#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
"""

import matplotlib.pyplot as plt
import numpy as np
from connected_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pymc3 as pm
import theano.tensor as tt

import pickle
from preparedata import preparation
from multiprocessing import Pool
import os
import sys
np.random.seed(1234567284)
path_results = '../../../../results/conn/'

    
pathrun = 'debug'

model_type = 'LF'

stations  = ['UWD']
date = '07-03-2018'
flaglocation = 'N'      # This is the flag for locations priors, F for Uniform, N for Normal

def parameters_init(mt):
    #Initialize parameters
    dxmax = 10.0
    dxmin = 4.0

    Ncycmin = 20
    Ncycmax = 60

    if mt == 'UF':
        bounds = np.array([[+8,10],[+9,+12],[+7,+11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20],[0.5,2.0]])
        boundsLoc = np.array([[-1000,1000],[1200,2500],[500,1500],[-1500,+1500],[-1500,2500],[2000,4000]])
    elif mt == 'LF':                                                 
        bounds = np.array([[+8,+12],[+8,+10],[+8,+11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20]])
        boundsLoc = np.array([[-1000,+1000],[-1000,1000],[2000,5000],[-500,0],[1500,2000],[500,1300]])

    bndtiltconst = 10000
    bndGPSconst = 200
    bnddeltaP0 = 1e+7
    tiltErr = 1e-5
    GPSErr = 1e-2
    locErrFact = 5
    
    a_parameter = 5
    thin = 10
    nwalkers = 64

    g = 9.8
    rho = 2700
    #chamber and elasticity
    #Elastic properties crust
    poisson = 0.25
    lame = 3e+9
    lamesar = 1e+9 
    const = -9. /(4 * 3.14) * (1 - poisson) / lame
    #Position of the chamber
    #Piston geometrical parameters
    Rcyl = 7e+2
    S = 3.14 * Rcyl**2
    #Conduit Properties
    mu = 1e+2
    ls = 4e+4
    ld = 2.5e+3
    rhog = rho * g
    S = 3.14 * Rcyl**2 
    rhog = rho * g
    ndim = len(bounds)

    return bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bnddeltaP0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog

def get_trace(path_trace,rhog,cs):
    path_results = '../../../../results/'
    data = pickle.load(open(path_results + 'data2ch.pickle','rb'))
    dtslip,tiltsty,tiltstx,tiltsly,tiltslx,gps,stackx,stacky,tstack,xsh,ysh,dsh,xshErr,yshErr,dshErr,strsrc,strsrcErr = data
    n = np.arange(1,len(tiltsty)+ 1 )
    tilt_std = 1e-5
    dt_std = 3600
    gps_std = 1
    with pm.Model() as model:
        gpsconst = pm.Uniform('gpsconst',lower = -15,upper = 15)
        A_mod = pm.Uniform('A_mod',lower = 0, upper = 1e+2)
        B_mod = pm.Uniform('B_mod',lower = 0, upper = 1e+2)
        C_mod = pm.Uniform('C_mod',lower = 0, upper = 1e+2)
        D_mod = pm.Uniform('D_mod',lower = 0, upper = 1e+2)
        E_mod = pm.Uniform('E_mod',lower = 0, upper = 1e+2)
        F_mod = pm.Uniform('F_mod',lower = 0, upper = 1e+2)
    
        Vd_exp = pm.Uniform('Vd_exp',lower = 8,upper = 11)
        Vd_mod = pm.Deterministic('Vd_mod',10**Vd_exp)
    
        kd_exp = pm.Uniform('kd_exp',lower = 7, upper = 10)
        kd_mod = pm.Deterministic('kd_mod',10**kd_exp)
        R1 = pm.Deterministic('R1',rhog * Vd_mod /(kd_mod * S)  )
        ratio = pm.Uniform('ratio',lower = 30 * 4 * R1 / (1 + R1), upper = 100 * 4 * R1 / (1 + R1))
        pspd_mod = pm.Uniform('pspd_mod',lower=1e+5,upper=1e+7)
        ptps_mod = pm.Deterministic('ptps_mod',ratio * pspd_mod)
        conds_mod = pm.Uniform('conds_mod',lower = 1, upper = 10)
    
        deltap_mod = pm.Uniform('deltap',lower = 1e+5,upper = ptps_mod )
        strsrc_mod = pm.Normal('strsrc_Mod',mu = strsrc, sigma = strsrcErr)
        Vs_mod = pm.Deterministic('Vs_mod',strsrc_mod / deltap_mod)
    
        
        #conds_mod = pm.Uniform('conds_mod',lower=1,upper=10)
        condd_mod = pm.Uniform('condd_mod',lower=1,upper=30)
        dsh_mod = pm.Normal('dsh_mod',mu = dsh, sigma = dshErr)
        xsh_mod = pm.Normal('xsh_mod',mu = xsh, sigma = xshErr)
        ysh_mod = pm.Normal('ysh_mod',mu = ysh, sigma = yshErr)
        coeffx = cs * dsh_mod * (x -  xsh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vd_mod
        coeffy = cs * dsh_mod * (y -  ysh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vd_mod
        tau2 = 8 * mu *ld * Vs_mod/ (3.14 * condd_mod**4 * kd_mod)    #Model set-up
        x_mod =gpsconst+ 4 * R1 / rhog * pspd_mod / (1 + R1) * n
        
        pslip_mod = -rhog * x_mod
        pstick_mod = pslip_mod + 4 * pspd_mod/ (1 + R1)
    
        
        trace2 = pm.load_trace(path_results + 'trace300000_strsrc_LF')
    return  trace2
    



if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]


pathtrunk = pathtrunk + '/' + date + '/'

pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
pathgg = path_results + pathgg


if not os.path.exists(pathgg):
    os.makedirs(pathgg)
    descr =input('Briefly describe the run: ')
    f =  open(pathgg + 'description.txt','w')
    f.write(descr)
    f.close()
else:
    print('Directory exists! CHeck that everything is ok!')
    
f = open(path_results+'dir_current_run.txt','w')
f.write(os.path.abspath(pathgg))
f.close()
path_trace = 'trace300000_strsrc_LF'
if  os.path.isfile(pathgg + 'progress.h5'):
    Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
    a = pickle.load(open(pathgg + 'parameters.pickle','rb'))
    if len(a)== 17:
        bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
        boundsLoc = 0
    elif len(a)== 18:
        bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
    reader = emcee.backends.HDFBackend(pathgg + 'progress.h5', read_only=True)
    niter  = reader.iteration
    print('The current run is: ',os.path.abspath(pathgg))
    answer = input('Found past run with ' + str(niter) +' samples. Do you want to continue this run? Enter the number of iteration that additionally you want to do: ')
    moreiter = int(answer)
    print('Restarting!!!!')
    nwalkers,ndim = reader.shape
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    move = emcee.moves.StretchMove(a=a_parameter)
    with Pool() as pool:
        if model_type == 'UF':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                            args=(x,y,
                                                  ls,ld,mu,
                                                  rhog,const,S,
                                                  tTilt,tGPS,tx,ty,GPS,
                                                  tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation),moves = [move], backend = backend, pool = pool)
        elif model_type =='LF':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_LF,
                                            args=(x,y,
                                                  ls,ld,mu,
                                                  rhog,const,S,
                                                  tTilt,tGPS,tx,ty,GPS,
                                                  tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation),moves = [move], backend = backend, pool = pool)
        sampler.run_mcmc(None, moreiter, progress=True,thin = thin)



else:
    print('The put of the current run is: ',pathgg)
    niter = input('How many iteration you want to run? ')
    bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = parameters_init(model_type)
    trace = get_trace(path_trace,rhog,const)
    Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0 = preparation(stations,date,locErrFact,model_type)
    pos,nwalkers,ndim = walkers_init_frompymc(trace,nwalkers,ndim,bounds,boundsLoc,rhog,S,locTruth,locErr,bndtiltconst,bndGPSconst,bndp0,Nst,model_type,flaglocation)
    pickle.dump((Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0),open(pathgg + 'data.pickle','wb'))
    pickle.dump((bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog),open(pathgg + 'parameters.pickle','wb'))
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    move = emcee.moves.StretchMove(a=a_parameter)
    with Pool() as pool:
        if model_type == 'UF':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                        args=(x,y,
                                            ls,ld,mu,
                                            rhog,const,S,
                                            tTilt,tGPS,tx,ty,GPS,
                                            tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation),moves = [move], backend = backend, pool = pool)
        elif model_type == 'LF':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_LF,
                                        args=(x,y,
                                            ls,ld,mu,
                                            rhog,const,S,
                                            tTilt,tGPS,tx,ty,GPS,
                                            tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation,flaglocation),moves = [move], backend = backend, pool = pool)
       
        print('Running main sampling')
        sampler.run_mcmc(pos,niter, progress = True,thin = thin)
        