#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:11:28 2020

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
from main_lib_UF import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from preparedata_UF import preparation_UF
from multiprocessing import Pool
import os
import sys
import shutil
np.random.seed(1234567284)
path_results = '../../../../../results/'

    
pathrun = 'debug_priors'
stations  = ['UWD']
date = '07-03-2018'
model_type = 'UF'

def parameters_init():
    #Initialize parameters
    dxmax = 5.0
    dxmin = 3.0

    Ncycmin = 20
    Ncycmax = 60
    bounds = np.array([[+8,+11],[+8,+11],[+8,+10],[dxmin,dxmax],[Ncycmin,Ncycmax],[1,10],[1,10]])
    bndtiltconst = 2000 
    bndGPSconst = 200
    bnddeltaP0 = 4e+7
    tiltErr = 1e-5
    GPSErr = 1e-2
    locErrFact = 5
    
    a_parameter = 4
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
    return bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bnddeltaP0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog






if len(stations) > 1:
    pathtrunk = model_type + '/' + str(len(stations)) + 'st'
else:
    pathrunk = model_type + '/' + stations[0]


pathtrunk = pathrunk + '/' + date + '/'

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
    bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = pickle.load(open(pathgg + 'parameters.pickle','rb'))
    reader = emcee.backends.HDFBackend(pathgg + 'progress.h5', read_only=True)
    niter  = reader.iteration
    answer = input('Found past run with ' + str(niter) +' samples. Do you want to continue this run? Enter the number of iteration that additionally you want to do: ')
    moreiter = int(answer)
    print('Restarting!!!!')
    nwalkers,ndim = reader.shape
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    move = emcee.moves.StretchMove(a=a_parameter)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                       args=(x,y,
                                        ls,ld,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndGPSconst,locTruth,locErr,nstation), backend = backend, pool = pool)
        sampler.run_mcmc(None, moreiter, progress=True,thin = thin)



else:
    niter = input('How many iteration you want to run? ')
    bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = parameters_init()
    Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0 = preparation_UF(stations,date,locErrFact)
    pos,nwalkers,ndim = walkers_init(nwalkers,ndim,bounds,rhog,S,locTruth,locErr,bndtiltconst,bndGPSconst,bndp0,Nst)
    pickle.dump((Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0),open(pathgg + 'data.pickle','wb'))
    pickle.dump((bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog),open(pathgg + 'parameters.pickle','wb'))
    filename = pathgg + "progress.h5"
    backend = emcee.backends.HDFBackend(filename)
    
    move = emcee.moves.StretchMove(a=a_parameter)
    
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
                                        args=(x,y,
                                            ls,ld,mu,
                                            rhog,const,S,
                                            tTilt,tGPS,tx,ty,GPS,
                                            tiltErr,GPSErr,bounds,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation), backend = backend, pool = pool)
       
        print('Running main sampling')
        sampler.run_mcmc(pos,niter, progress = True,thin = thin)
        