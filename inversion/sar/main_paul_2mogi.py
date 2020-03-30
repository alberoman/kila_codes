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
pathgg_results = 'results/2mogi/'
pathgg_data = '../../data/sar/'
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
data = prepare_data(ref_coord,pathgg_data,filename)
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
    strExp1 = pm.Uniform('strExp1',lower = 12, upper = 18)
    strExp2 = pm.Uniform('strExp2',lower = 12, upper = 18)
    strSource1 = pm.Deterministic('strSource1',-10**strExp1)
    strSource2 = pm.Deterministic('strSource2',-10**strExp2)

    depthSource1 = pm.Uniform('depthSource1',lower = 500, upper = 2500)
    depthSource2 = pm.Uniform('depthSource2',lower = 500, upper = 5000)
    xSource1 = pm.Uniform('xSource1', lower = -10000 ,upper = 10000)
    xSource2 = pm.Uniform('xSource2', lower = -10000 ,upper = 10000)
    ySource1 = pm.Uniform('ySource1', lower = -10000, upper =  10000)
    ySource2 = pm.Uniform('ySource2', lower = -10000, upper =  10000)
    
    R_asc1 = tt.sqrt((x_asc - xSource1)**2 + (y_asc - ySource1)**2 + depthSource1 **2)
    R_asc2 = tt.sqrt((x_asc - xSource2)**2 + (y_asc - ySource2)**2 + depthSource2 **2)
    
    coeff_east_asc1 = (x_asc - xSource1)/ R_asc1**3 * elastic_constant
    coeff_east_asc2 = (x_asc - xSource2)/ R_asc2**3 * elastic_constant
    coeff_north_asc1 = (y_asc - ySource1)/ R_asc1**3 * elastic_constant
    coeff_north_asc2 = (y_asc - ySource2)/ R_asc2**3 * elastic_constant
    coeff_up_asc1 = depthSource1 / R_asc1**3 * elastic_constant
    coeff_up_asc2 = depthSource2 / R_asc2**3 * elastic_constant
    
    east_asc1 = strSource1 * coeff_east_asc1
    east_asc2 = strSource2 * coeff_east_asc2
    north_asc1 = strSource1 * coeff_north_asc1
    north_asc2 = strSource2 * coeff_north_asc2
    vert_asc1 = strSource1 * coeff_up_asc1
    vert_asc2 = strSource2 * coeff_up_asc2
    
    east_asc = east_asc1 + east_asc2
    north_asc = north_asc1 + north_asc2
    vert_asc = vert_asc1 + vert_asc2
    
    Ulos_asc = east_asc * UEast_asc + north_asc * UNorth_asc + vert_asc * UVert_asc
    
    R_des1 = tt.sqrt((x_des - xSource1)**2 + (y_des - ySource1)**2 + depthSource1 **2)
    R_des2 = tt.sqrt((x_des - xSource2)**2 + (y_des - ySource2)**2 + depthSource2 **2)
    
    coeff_east_des1 = (x_des - xSource1)/ R_des1**3 * elastic_constant
    coeff_east_des2 = (x_des - xSource2)/ R_des2**3 * elastic_constant
    coeff_north_des1 = (y_des - ySource1)/ R_des1**3 * elastic_constant
    coeff_north_des2 = (y_des - ySource2)/ R_des2**3 * elastic_constant
    coeff_up_des1 = depthSource1 / R_des1**3 * elastic_constant
    coeff_up_des2 = depthSource2 / R_des2**3 * elastic_constant
    
    east_des1 = strSource1 * coeff_east_des1
    east_des2 = strSource2 * coeff_east_des2
    north_des1 = strSource1 * coeff_north_des1
    north_des2 = strSource2 * coeff_north_des2
    vert_des1 = strSource1 * coeff_up_des1
    vert_des2 = strSource2 * coeff_up_des2
    
    east_des = east_des1 + east_des2
    north_des = north_des1 + north_des2
    vert_des = vert_des1 + vert_des2
    
    Ulos_des = east_des * UEast_des + north_des * UNorth_des + vert_des * UVert_des
    
    UObs_asc = mv.MvNormal('Uobs_asc', mu = Ulos_asc , cov = covariance_asc, observed = U_asc)
    UObs_des = mv.MvNormal('Uobs_des', mu = Ulos_des , cov = covariance_des, observed = U_des)
    trace = pm.sample(100000,init='advi',tune=1000)
trace = trace[Nburn:] #Discard first 1000 samples of each chain  
print(pm.summary(trace))
map_estimate = pm.find_MAP(model=model)
print(map_estimate)

results = {}
results['MAP'] = map_estimate
results['trace'] = trace
results['ref_coord'] = data['ref_coord']
results['iterations'] = Niter
pickle.dump(results,open(pathgg_results + 'Mogi_Metropolis_' + str(Niter) + '_2mogi.pickle','wb'))









    
    
    
    
    
    
    





