#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
plt"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle 

def log_likelihood(parameters, tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,GPSObs,tiltErr,GPSErr):
    Vs,Vd,pt,conds,condd,taus,taud,ks = parameters

    tx1Mod,ty1Mod,tx2Mod,ty2Mod,GPSMod = DirectModelEmcee(tTilt,tGPS,Vs,Vd,pt,conds,condd,taus,taud,ks)
    sigma2Tilt = tiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx1 = 0.5 * np.sum((tx1Obs - tx1Mod) ** 2 / sigma2Tilt)
    likety1 = 0.5 * np.sum((ty1Obs - ty1Mod) ** 2 / sigma2Tilt)
    liketx2 = 0.5 * np.sum((tx2Obs - tx2Mod) ** 2 / sigma2Tilt)
    likety2 = 0.5 * np.sum((ty2Obs - ty2Mod) ** 2 / sigma2Tilt)
    likeGPS = 0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS)
    return -liketx1 -likety1 - liketx2 -likety2 - likeGPS 

def log_prior(parameters,bounds):
   Vs,Vd,pt,conds,condd,taus,taud,ks= parameters
   if bounds[0,0] < Vs < bounds[0,1] and  bounds[1,0] < Vd < bounds[1,1] and bounds[2,0] < pt < bounds[2,1] and bounds[3,0] < conds < bounds[3,1] and bounds[4,0] < condd < bounds[4,1] and bounds[5,0] < taus < bounds[5,1] and bounds[6,0] < taud < bounds[6,1] and bounds[7,0] < ks < bounds[7,1] :
       return 0.0
   return -np.inf

def log_probability(param, tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,GPSObs,tiltErr,GPSErr):
    bounds = np.array([[1e+8,1e+10],[1e+9,1e+11],[5e+6,1e+8],[1,10],[1,10],[1e+6,2e+7],[1e+5,1e+7],[1e+8,1e+11]]) #Vs,Vd,ks,kd
    lp = log_prior(param,bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param,tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,GPSObs,tiltErr,GPSErr)



    
xSource = np.array([0,0]) #The first value here corresponds to the shallow source (the one feeding)
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = 100
ystation2 = -1000
sigma_tilt = 0.2
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 3e+4
ld = 2.5e+3
condd = 5
conds = 4
#Sources volumes and compressibilities
Vs = 2e+10
Vd  = 1e+9
ks = 1e+9
kd = 1e+9
#Pressures and friction6
taus = 5e+6
taud = 3e+6
mu = 100
#Pressures and friction
pt = 20e+6
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
tiltPert = 0.05
cs = -9. /(4 * 3.14) * (1 - poisson) / lame

tiltErr = 1e-5
GPSErr = 1e-2
coeffxs1 = cs * depthSource[0] * (xstation1 -  xSource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffys1 = cs * depthSource[0] * (ystation1 -  ySource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffxd1 = cs * depthSource[1] * (xstation1 -  xSource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)
coeffyd1 = cs * depthSource[1] * (ystation1 -  ySource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)

coeffxs2 = cs * depthSource[0] * (xstation2 -  xSource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffys2 = cs * depthSource[0] * (ystation2 -  ySource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffxd2 = cs * depthSource[1] * (xstation2 -  xSource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)
coeffyd2 = cs * depthSource[1] * (ystation2 -  ySource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)

tTilt,tx1,ty1,tx2,ty2,GPS,tGPS,Nmax = Twochambers_syntethic_LF(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,conds,condd,Vs,Vd,ks,kd,taus,taud,mu,pt,Rcyl,rho,g,poisson,lame,tiltPert)

tx1Mod,ty1Mod,tx2Mod,ty2Mod,GPSmod = DirectModelEmcee(tTilt,tGPS,Vs,Vd,pt,conds,condd,taus,taud,ks)
#plt.plot(tTilt,tx1,tTilt,tx1Mod)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([Vs,Vd,pt,conds,condd,taus,taud,ks])
initial = initial +   0.5 * initial *np.random.randn(len(initial))

soln = minimize(nll, initial, args=(tTilt,tGPS,tx1,ty1,tx2,ty2,GPS,tiltErr,GPSErr))


pos = soln.x + 0.5 * soln.x * np.random.randn(32, len(initial))
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(tTilt,tGPS,tx1,ty1,tx2,ty2,GPS,tiltErr,GPSErr))
sampler.run_mcmc(pos, 100, progress=True)
samples = sampler.get_chain()
samplesFlat = sampler.get_chain(flat = True)
with open('samples.pickle','wb') as filen:
    pickle.dump(samples,samplesFlat,filen)




    
