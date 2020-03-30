#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:33:16 2019

@author: aroman
"""

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
lat_0 = 2500
lon_0 =  -1000
depth_0 = 3000
source_strength = 1e+15
latitude = np.linspace(-10000,10000,100)
longitude = np.linspace(-10000,10000,100)
LON,LAT = np.meshgrid(longitude,latitude)
poisson = 0.25
lame = 1e+9
elastic_constant = (1 - poisson) / lame
R = np.sqrt((LON - lon_0)**2 + (LAT - lat_0)**2 + depth_0 **2)
coeff_east = (LON - lon_0)/ R**3 * elastic_constant
coeff_north = (latitude - lat_0)/ R**3 * elastic_constant
coeff_up = depth_0 / R**3 * elastic_constant
east_mod = source_strength * coeff_east
north_mod = source_strength * coeff_north
up_mod = source_strength * coeff_up
noise_coeff = 0.1
sigma_def = np.max(up_mod) * noise_coeff
np.random.seed(123)
up_pert = up_mod + np.random.randn(np.shape(up_mod)[0],np.shape(up_mod)[1]) * sigma_def
LON_vector = np.squeeze(LON.reshape(1,LON.size))
LAT_vector = np.squeeze(LAT.reshape(1,LAT.size))
up_pert_vector = np.squeeze(up_pert.reshape(1,up_pert.size))
number_points = 10
#Put some distance constraints from the source
dist_max = 3000 # Distance in metres from the source
R = np.sqrt((LON_vector - lon_0)**2 + (LAT_vector - lat_0)**2)
#LON_vector = LON_vector[R < dist_max]
#LAT_vector = LAT_vector[R < dist_max]
#up_pert_vector = up_pert_vector[R < dist_max]
#ind = np.random.randint(low = 1, high = len(LON_vector),size = number_points)
#LON_vector = LON_vector[ind]
#LAT_vector = LAT_vector[ind]
#up_pert_vector = up_pert_vector[ind]
with pm.Model() as model:
    source_strength_exp = pm.Uniform('source_strength_exp',lower = 12, upper = 18)
    source_str = pm.Deterministic('source_str',10**source_strength_exp)
    lon = pm.Uniform('lon',lower = - 5000 ,upper =  5000)
    lat = pm.Uniform('lat',lower =  -5000 ,upper =  5000)
    depth = pm.Uniform('depth',lower = 1000, upper = 5000)
    R = np.sqrt((LON_vector - lon)**2 + (LAT_vector - lat)**2 + depth**2)
    coeff_up = depth / R**3 * elastic_constant
    up_mod = source_str * coeff_up
    up_obs = pm.Normal('up_obs', mu = up_mod, sigma = sigma_def, observed = up_pert_vector)
    trace = pm.sample(10000)
    