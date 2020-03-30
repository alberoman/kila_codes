#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:16:36 2019

@author: aroman
"""
from coord_conversion import *
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pickle
from skgstat import Variogram
step_down = 5
filename = '../data/sar/S1-GUNW-D-R-087-tops-20180622_20180529-161541-20480N_18471N-PP-652b-v2_0_1.nc'
rootgrp = Dataset(filename,'r')
data= rootgrp['science/grids/data/']
geometry = rootgrp['science/grids/imagingGeometry/']
wavelength = np.float64(rootgrp['science/radarMetaData'].variables['wavelength'][:])
lonMeta = geometry.variables['longitudeMeta'][:]
latMeta = geometry.variables['latitudeMeta'][:]
heightsMeta = geometry.variables['heightsMeta'][:]
heightsMeta = heightsMeta[:]
incidence = geometry.variables['incidenceAngle'][:]
look = geometry.variables['lookAngle'][:]
azimuth = geometry.variables['azimuthAngle'][:]
lon_sar = data.variables['longitude'][:]
lat_sar = data.variables['latitude'][:]
phase = data.variables['unwrappedPhase'][:]
coh = data.variables['coherence'][:]
phase_downsamp = phase[::step_down,::step_down]
lat_sar_downsamp = lat_sar[::step_down]
lon_sar_downsamp = lon_sar[::step_down]
lon = lon_sar_downsamp
lat = lat_sar_downsamp

phase = phase_downsamp
rootgrp.close()
x_luc = 300
y_luc = 180
x_rlc =  400
y_rlc = 240
#x_luc = input('Enter the x coodinate of the left upper corner:')
#y_luc = input('Enter the y coodinate of the left upper corner:')
#x_rlc = input('Enter the x coodinate of the right lower corner:')
#y_rlc = input('Enter the y coodinate of the right lower corner:')
fig,ax = plt.subplots(nrows = 1, ncols=2)
phase_original = phase[:]
phase_original = phase_original.reshape(1,phase_original.size)
LON,LAT = np.meshgrid(lon,lat)
lon_original = LON.reshape(1,LON.size)
lat_original = LAT.reshape(1,LAT.size)
#ax[0].scatter(lon_original,lat_original,c=np.mod(phase_original,wavelength/2))

#Cropping
phase = phase[y_luc:y_rlc,x_luc:x_rlc] 
lon = lon[x_luc:x_rlc]
lat = lat[y_luc:y_rlc]
ref_point = [0.5 * (lon[0] + lon[-1]), 0.5 * (lat[0] + lat[-1])]
LON,LAT = np.meshgrid(lon,lat)
lon = LON.reshape(1,LON.size)
lat = LAT.reshape(1,LAT.size)
phase = phase.reshape(1,phase.size)
c1 = ax[0].scatter(lon,lat,c=np.mod(phase,wavelength/2))
plt.colorbar(c1)
sll = np.concatenate((lon,lat),0)
xy = llh2local(sll,ref_point)
xy = xy*1000
A = np.concatenate((xy.transpose(),np.ones((lat.size,1))),1)
B = np.linalg.lstsq(A,np.transpose(phase))[0]
C = np.dot(A,B)
phase_detrend = phase - C.transpose()
c2 = ax[1].scatter(lon,lat,c=np.mod(phase_detrend,wavelength/2))
plt.colorbar(c2)
plt.savefig('detrend_comparison.pngn')



