#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:38:32 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:47:26 2020

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
from onech_lib import *
import numpy as np
import emcee
import pickle
from preparedata import preparation
from multiprocessing import Pool
import os
import sys
np.random.seed(1234567284)
path_results = '../../../../results/onech/'

    

pathrun = 'alpha'

model_type = 'LF'

stations  = ['UWD']
date = '07-03-2018'
flaglocation = 'N'      # This is the flag for locations priors, F for Uniform, N for Normal

def parameters_init(mt):
    #Initialize parameters
    dxmax = 5.0
    dxmin = 3.0

    Ncycmin = 20
    Ncycmax = 60

    if mt == 'UF':
        bounds = np.array([[+8,+11],[+8,+11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[0.5,2]])
        boundsLoc = np.array([[-1000,1000],[1200,2500],[500,1500],[-1500,+1500],[-1500,2500],[2000,4000]])
    elif mt == 'LF':                                                 
        bounds = np.array([[+8,+11],[+8,+11],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[0.5,2]])
        boundsLoc = np.array([[-1000,+1000],[-1000,1000],[2000,5000],[-500,0],[1500,2000],[500,1300]])

    bndtiltconst = 2000
    bndGPSconst = 70
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
    return bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,locErrFact,a_parameter,thin,nwalkers,ls,mu,ndim,const,S,rhog






if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]


pathtrunk = pathtrunk + '/' + date + '/'

pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
pathgg = path_results + pathgg

f = open(path_results+'dir_current_run.txt','w')
f.write(os.path.abspath(pathgg))
f.close()
if  os.path.isfile(pathgg + 'progress.h5'):
    Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
    a = pickle.load(open(pathgg + 'parameters.pickle','rb'))
    if len(a)== 17:
        bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,locErrFact,a_parameter,thin,nwalkers,ls,mu,ndim,const,S,rhog = a
        boundsLoc = 0
    elif len(a)== 18:
        bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,locErrFact,a_parameter,thin,nwalkers,ls,mu,ndim,const,S,rhog = a
    reader = emcee.backends.HDFBackend(pathgg + 'progress.h5', read_only=True)
    niter  = reader.iteration
    print('The current run is: ',os.path.abspath(pathgg))
    answer = input('Found past run with ' + str(niter) +' samples. Do you want to continue this run? Enter the number of iteration that additionally you want to do: ')
    moreiter = int(answer)
    nwalkers,ndim = reader.shape
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    move = emcee.moves.StretchMove(a=a_parameter)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_onech,
                                        args=(x,y,
                                              ls,mu,
                                              rhog,const,S,
                                              tTilt,tGPS,tx,ty,GPS,
                                              tiltErr,GPSErr,bounds,boundsLoc,bndGPSconst,bndtiltconst,locTruth,locErr,nstation,flaglocation),moves = [move], backend = backend, pool = pool)
