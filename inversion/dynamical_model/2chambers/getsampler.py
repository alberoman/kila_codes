#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:46:38 2020

@author: aroman
"""

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
import pickle
from preparedata import preparation
from multiprocessing import Pool
import os
import sys
np.random.seed(1234567284)
path_results = '../../../../results/conn/'

    

pathrun = 'alpha'

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
        bounds = np.array([[+8,10],[+9,+12],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20],[0.5,2.0]])
        boundsLoc = np.array([[-1000,1000],[1200,2500],[500,1500],[-1500,+1500],[-1500,2500],[2000,4000]])
    elif mt == 'LF':                                                 
        bounds = np.array([[+8,+12],[+8,+10],[+8,+11],[8,11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,20],[0.5,2.0]])
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






if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]


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
    answer = print('Found past run with ' + str(niter) +' samples)

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
  