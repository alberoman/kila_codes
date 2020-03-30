#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:47:56 2019
Main Inversion code
@author: aroman
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import theano
import pickle
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import theano.tensor as tt

import pymc3 as pm
import pymc3.distributions.multivariate as mv
from prepare_insar_paul import prepare_data
pathgg_results = 'results/'
pathgg_data = '../../data/sar/'
fname = 'S1-20180622_20180529_coh_0.5.pickle'
Niter = 100000 # Number of iterations
Nburn = 1000 #Burn-in phase, samples to disregard
if Nburn >= Niter:
    print('Error:Nburn greater than the number of iterations!!! Decrease the number of burn-in phase or increase iterations')
    sys.exit()
ref_coord = np.array([-155.274,19.395]) #Cordinate of CRIM station
dg2rd = np.pi / 180

#Staff to be used as parameters
poisson = 0.25
lame = 1e+9
elastic_constant = (1 - poisson) / lame

#Preparing data
print('Preparing data')
filename = '_t124.out'
data = prepare_data(ref_coord,pathgg_data,filename)
x_asc = data['obs'][:,0]
y_asc = data['obs'][:,1]
lon_asc = data['LonLat'][:,0]
lat_asc = data['LonLat'][:,1]
covariance_asc = data['cov']
inc_asc = data['inc'] * dg2rd
heading_asc = data['heading'] * dg2rd
UEast_asc = -np.cos(heading_asc) * np.sin(inc_asc) 
UNorth_asc = np.sin(heading_asc) * np.sin(inc_asc) 
UVert_asc = np.cos(inc_asc)
U_asc = data['vel']
#Setting some bounds
xmin_asc = np.min(x_asc)
xmax_asc = np.max(x_asc)
ymin_asc = np.min(y_asc)
ymax_asc = np.max(y_asc)

filename = '_t87.out'
data = prepare_data(ref_coord,filename)
x_des = data['obs'][:,0]
y_des = data['obs'][:,1]
lon_des = data['LonLat'][:,0]
lat_des = data['LonLat'][:,1]
covariance_des = data['cov']
inc_des = data['inc'] * dg2rd
heading_des = data['heading'] * dg2rd
UEast_des = -np.cos(heading_des) * np.sin(inc_des) 
UNorth_des = np.sin(heading_des) * np.sin(inc_des) 
UVert_des = np.cos(inc_des)
U_des = data['vel']
#Setting some bounds
xmin_des = np.min(x_des)
xmax_des = np.max(x_des)
ymin_des = np.min(y_des)
ymax_des = np.max(y_des)

print('Starting inversion')
#Setup inversion
with pm.Model() as model:
    strExp = pm.Uniform('strExp',lower = 12, upper = 18)
    strSource = pm.Deterministic('strSource',-10**strExp)
    depthSource = pm.Uniform('depthSource',lower = 1000, upper = 5000)
    xSource = pm.Uniform('xSource', lower = -10000 ,upper = 10000)
    ySource = pm.Uniform('ySource', lower = -10000, upper =  10000)
    R_asc = tt.sqrt((x_asc - xSource)**2 + (y_asc - ySource)**2 + depthSource **2)
    coeff_east_asc = (x_asc - xSource)/ R_asc**3 * elastic_constant
    coeff_north_asc = (y_asc - ySource)/ R_asc**3 * elastic_constant
    coeff_up_asc = depthSource / R_asc**3 * elastic_constant
    east_asc = strSource * coeff_east_asc
    north_asc = strSource * coeff_north_asc
    vert_asc = strSource * coeff_up_asc
    Ulos_asc = east_asc * UEast_asc + north_asc * UNorth_asc + vert_asc * UVert_asc
    
    R_des = tt.sqrt((x_des - xSource)**2 + (y_des - ySource)**2 + depthSource **2)
    coeff_east_des = (x_des - xSource)/ R_des**3 * elastic_constant
    coeff_north_des = (y_des - ySource)/ R_des**3 * elastic_constant
    coeff_up_des = depthSource / R_des**3 * elastic_constant
    east_des = strSource * coeff_east_des
    north_des = strSource * coeff_north_des
    vert_des = strSource * coeff_up_des
    Ulos_des = east_des * UEast_des + north_des * UNorth_des + vert_des * UVert_des
    
    
    UObs_asc = mv.MvNormal('Uobs_asc', mu = Ulos_asc , cov = covariance_asc, observed = U_asc)
    UObs_des = mv.MvNormal('Uobs_des', mu = Ulos_des , cov = covariance_des, observed = U_des)
    step = pm.Metropolis()
    trace = pm.sample(Niter,step)
trace = trace[Nburn:] #Discard first 1000 samples of each chain  
print(pm.summary(trace))
map_estimate = pm.find_MAP(model=model)
print(map_estimate)

results = {}
results['MAP'] = map_estimate
results['trace'] = trace
results['ref_coord'] = data['ref_coord']
results['iterations'] = Niter
pickle.dump(results,open(pathgg_results + 'Mogi_Metropolis_' + str(Niter) + '_coh_0.5.pickle','wb'))









    
    
    
    
    
    
    





