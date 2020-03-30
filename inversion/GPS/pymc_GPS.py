#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:22:01 2019

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:17:50 2019

@author: aroman
"""
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import pickle
import matplotlib.pyplot as plt

def vectorize_GPS(data):
    east = []
    north = []
    up  = []
    lon = []
    lat = []
    station_name = []
    for station in data:
        station_name.append(station)
        east.append(data[station]['east'][-1] - data[station]['east'][0])
        north.append(data[station]['north'][-1] - data[station]['north'][0])
        up.append(data[station]['up'][-1] - data[station]['up'][0])
        lon.append(data[station]['lon_meters'])
        lat.append(data[station]['lat_meters'])
    return station_name,np.array(east),np.array(north),np.array(up),np.array(lon),np.array(lat)

pathgg_GPS = '../../GPS/nevada/'
GPS = pickle.load(open(pathgg_GPS + 'dict_GPS.pickle','rb'))
stations,east_disp,north_disp,up_disp,longitude,latitude = vectorize_GPS(GPS)
poisson = 0.25
lame = 1e+9
elastic_constant = (1 - poisson) / lame
#Setup inversion
with pm.Model() as model:
    source_strength_exp = pm.Uniform('source_strength_exp',lower = 12, upper = 18)
    source_strength = pm.Deterministic('source_strength',10**source_strength_exp)
    lon_0 = pm.Uniform('lon_0',lower = np.min(longitude) - 3000 ,upper = np.max(longitude) + 3000)
    lat_0 = pm.Uniform('lat_0',lower = np.min(latitude) - 3000 ,upper = np.max(latitude) + 3000)
    depth_0 = pm.Uniform('depth',lower = 1000, upper = 5000)
    R = tt.sqrt((longitude - lon_0)**2 + (latitude - lat_0)**2 + depth_0 **2)
    coeff_east = (longitude - lon_0)/ R**3 * elastic_constant
    coeff_north = (latitude - lat_0)/ R**3 * elastic_constant
    coeff_up = depth_0 / R**3 * elastic_constant
    east_mod = source_strength * coeff_east
    north_mod = source_strength * coeff_north
    up_mod = source_strength * coeff_up
#    east_obs = pm.Normal('east_obs' , mu = east_mod, sigma = 0.01, observed = east_disp )
#    north_obs = pm.Normal('north_obs' , mu = north_mod, sigma = 0.01, observed = north_disp )
    up_obs = pm.Normal('up_obs' , mu = up_mod, sigma = 0.01, observed = up_disp )

    trace = pm.sample(10000,tune=500)