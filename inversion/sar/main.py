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
from prepare_insar import prepare_data
pathgg_results = 'results/'
pathgg_data = '../../data/sar/'
fname = 'S1-20180622_20180529_coh_0.5.pickle'
Niter = 100000 # Number of iterations
Nburn = 1000 #Burn-in phase, samples to disregard
if Nburn >= Niter:
    print('Error:Nburn greater than the number of iterations!!! Decrease the number of burn-in phase or increase iterations')
    sys.exit()
ref_coord = np.array([-155.274,19.395]) #Cordinate of CRIM station

#Preparing data
print('Preparing data')
data = prepare_data(pathgg_data,fname,ref_coord)
print('Starting inversion')

poisson = 0.25
lame = 1e+9
elastic_constant = (1 - poisson) / lame

x = data['obs'][:,0]
y = data['obs'][:,1]
lon = data['LonLat'][:,0]
lat = data['LonLat'][:,1]
covariance = data['cov']
dg2rd = np.pi / 180
inc = data['inc'] * dg2rd
azi = data['azi'] * dg2rd
UEast = np.cos(azi) * np.sin(inc)  # East component of LOS
UNorth = np.sin(azi) * np.sin(inc); # North unit vector
UVert = np.cos(inc) # Vertical unit vector
U = data['los']

#Setting some bounds
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)

#Setup inversion
with pm.Model() as model:
    strExp = pm.Uniform('strExp',lower = 12, upper = 18)
    strSource = pm.Deterministic('strSource',-10**strExp)
    xSource = pm.Uniform('xSource', lower = xmin  ,upper = xmax)
    ySource = pm.Uniform('ySource', lower = ymin,upper =  ymax)
    depthSource = pm.Uniform('depthSource',lower = 1000, upper = 5000)
    R = tt.sqrt((x - xSource)**2 + (y - ySource)**2 + depthSource **2)
    coeff_east = (x - xSource)/ R**3 * elastic_constant
    coeff_north = (y - ySource)/ R**3 * elastic_constant
    coeff_up = depthSource / R**3 * elastic_constant
    east = strSource * coeff_east
    north = strSource * coeff_north
    vert= strSource * coeff_up
    Ulos = east * UEast + north * UNorth + vert * UVert
    UObs = mv.MvNormal('Uobs', mu = Ulos , cov = covariance, observed = U)
    step = pm.Metropolis()
    trace = pm.sample(Niter,step)
trace = trace[Nburn:] #Discard first 1000 samples of each chain  
print(pm.summary(trace))
map_estimate = pm.find_MAP(model=model)
print(map_estimate)
R = np.sqrt((x - map_estimate['xSource'])**2 + 
                (y - map_estimate['ySource'])**2 
                + map_estimate['depthSource'] **2)

coeff_east = (x - map_estimate['xSource'])/ R**3 * elastic_constant
coeff_north = (y - map_estimate['ySource'])/ R**3 * elastic_constant
coeff_up = map_estimate['depthSource'] / R**3 * elastic_constant
east = map_estimate['strSource'] * coeff_east
north = map_estimate['strSource'] * coeff_north
vert= map_estimate['strSource'] * coeff_up
UMAP = east * UEast + north * UNorth + vert * UVert
Umax = np.max(U)
Umin = np.min(U)
fig,ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].scatter(x,y,c = U, vmin = Umin, vmax = Umax)
ax[0].plot(map_estimate['xSource'],map_estimate['ySource'],'ro')
ax[1].scatter(x,y,c = UMAP, vmin = Umin, vmax = Umax)
results = {}
results['MAP'] = map_estimate
results['trace'] = trace
results ['ULOS_measured'] = U
results['ULOS_MAP'] = UMAP
results['x'] = x
results['y'] = y
results['lon'] = lon
results['lat'] = lat
results['ref_coord'] = data['ref_coord']
results['iterations'] = Niter
pickle.dump(results,open(pathgg_results + 'Mogi_Metropolis_' + str(Niter) + '_coh_0.5.pickle','wb'))
plt.show()








    
    
    
    
    
    
    





