#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:26:33 2020

@author: aroman
"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
from multiprocessing import Pool
import os

def log_likelihood(parameters, tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,dtx1Obs,dty1Obs,dtx2Obs,dty2Obs,GPSObs,tiltErr,dtiltErr,GPSErr):
    Vs_exp,Vd_exp,k_exp,R5log,R3,conds,condd = parameters

    tx1Mod,ty1Mod,tx2Mod,ty2Mod,GPSMod,dtx1Mod,dty1Mod,dtx2Mod,dty2Mod, = DirectModelEmcee_inv(tTilt,tGPS,Vs_exp,Vd_exp,k_exp,R5log,R3,conds,condd)
    sigma2Tilt = tiltErr ** 2
    sigma2dTilt = dtiltErr ** 2
    sigma2GPS = GPSErr ** 2
    liketx1 = -0.5 * np.sum((tx1Obs - tx1Mod) ** 2 / sigma2Tilt )  #-len(tx1Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    likety1 = -0.5 * np.sum((ty1Obs - ty1Mod) ** 2 / sigma2Tilt ) # -len(ty1Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    liketx2 = -0.5 * np.sum((tx2Obs - tx2Mod) ** 2 / sigma2Tilt ) # -len(tx2Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    likety2 = -0.5 * np.sum((ty2Obs - ty2Mod) ** 2 / sigma2Tilt ) # -len(ty2Obs)/ 2 * np.log(6.28 * sigma2Tilt**2)
    
    likedtx1 = -0.5 * np.sum((dtx1Obs - dtx1Mod) ** 2 / sigma2dTilt )# -len(dtx1Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    likedty1 = -0.5 * np.sum((dty1Obs - dty1Mod) ** 2 / sigma2dTilt )# -len(dty1Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    likedtx2 = -0.5 * np.sum((dtx2Obs - dtx2Mod) ** 2 / sigma2dTilt )#-len(dtx2Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    likedty2 = -0.5 * np.sum((dty2Obs - dty2Mod) ** 2 / sigma2dTilt ) #-len(dty2Obs)/ 2 * np.log(6.28 * sigma2dTilt**2)
    
    likeGPS = -0.5 * np.sum((GPSObs - GPSMod) ** 2 / sigma2GPS) -len(GPSObs)/ 2 * np.log(6.28 * sigma2GPS**2)
    return +liketx1  + likety1 + liketx2 + likety2 +likedtx1 +likety1 +likedtx2 +likedty2# - likeGPS

def log_prior(parameters,bounds):
   Vs_exp,Vd_exp,k_exp,R5log,R3,conds, condd = parameters

   if bounds[0,0] < Vs_exp < bounds[0,1] and bounds[1,0] < Vd_exp < bounds[1,1] and bounds[2,0] < k_exp < bounds[2,1] and bounds[3,0] < R5log < bounds[3,1] and bounds[4,0] < R3< bounds[4,1] and bounds[5,0] < conds < bounds[5,1] and bounds[6,0] < condd < bounds[6,1]:
       return 0
   return -np.inf

def log_probability(param, tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,dtx1Obs,dty1Obs,dtx2Obs,dty2Obs,GPSObs,tiltErr,dtiltErr,GPSErr,bounds):
    
    lp = log_prior(param,bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param,tTilt,tGPS,tx1Obs,ty1Obs,tx2Obs,ty2Obs,dtx1Obs,dty1Obs,dtx2Obs,dty2Obs,GPSObs,tiltErr,dtiltErr,GPSErr)




    
xSource = np.array([0,0]) #The first value here corresponds to the shallow source (the one feeding)
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = 100
ystation2 = -4000
sigma_tilt = 0.2
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 3e+4
ld = 2.5e+3
condd = 4
conds = 4
#Sources volumes and compressibilities
Vd_exp = 9.698970004336019
Vs_exp = 10
k_exp = 9
Vd = 10**Vd_exp
Vs = 10**Vs_exp
ks = 10**k_exp
#Pressures and friction6
R3 = 3
R5 = 0.2
R5log = np.log10(R5)
taus = 7e+6
taud = taus * R5
mu = 100
#Pressures and friction
pt = taus * R3
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
tiltPert = 0.0
cs = -9. /(4 * 3.14) * (1 - poisson) / lame

tiltErr = 1e-5
dtiltErr = 1e-8
GPSErr = 1e-2
coeffxs1 = cs * depthSource[0] * (xstation1 -  xSource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffys1 = cs * depthSource[0] * (ystation1 -  ySource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffxd1 = cs * depthSource[1] * (xstation1 -  xSource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)
coeffyd1 = cs * depthSource[1] * (ystation1 -  ySource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)

coeffxs2 = cs * depthSource[0] * (xstation2 -  xSource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffys2 = cs * depthSource[0] * (ystation2 -  ySource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffxd2 = cs * depthSource[1] * (xstation2 -  xSource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)
coeffyd2 = cs * depthSource[1] * (ystation2 -  ySource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)

tTilt,tx1,ty1,tx2,ty2,GPS,tGPS,Nmax = Twochambers_syntethic_LF(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,conds,condd,Vs,Vd,ks,taus,taud,mu,pt,Rcyl,rho,g,poisson,lame,tiltPert)

tx1Mod,ty1Mod,tx2Mod,ty2Mod,GPSMod,dtx1Mod,dty1Mod,dtx2Mod,dty2Mod = DirectModelEmcee_inv(tTilt,tGPS,Vs_exp,Vd_exp,k_exp,R5log,R3,conds,condd)
dtx1 = np.diff(tx1) / np.diff(tTilt)
dty1 = np.diff(ty1) / np.diff(tTilt)
dtx2 = np.diff(tx2) / np.diff(tTilt)
dty2 = np.diff(ty2) / np.diff(tTilt)
plt.plot(tTilt[:-1],dtx1,tTilt[:-1],dtx1Mod)
initial = np.array([Vs_exp,Vd_exp,k_exp,R5log,R3,conds,condd])
bounds = np.array([[+9,+11],[+8,+10],[+8,+10],[-2,0],[1.1,5],[1,5],[1,5]]) #Vs,Vd,as,ad,taus,R5,R3,k_exp
log_likelihood(initial,tTilt,tGPS,tx1,ty1,tx2,ty2,dtx1,dty1,dtx2,dty2,GPS,tiltErr,dtiltErr,GPSErr)
