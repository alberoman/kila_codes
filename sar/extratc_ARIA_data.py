#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:21:26 2019

@author: aroman
"""
import math
from pyproj import Proj, transform
import numpy as np
from netCDF4 import Dataset
import gdal
from mpl_toolkits.basemap import Basemap
from scipy import interpolate
import pickle
from PCA_projection_tilt import *
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

def remove_value(x,y,val1,val2,par,cond):
    #Remove from vector val1 and its coordinate z,y entries that are in the val1 array according to par in the val2 array
    if cond == 'gt':
        x =  x[val2 > par]
        y = y[val2 > par]
        val1 = val1[val2 > par]
    else:
        x =  x[val2 < par]
        y = y[val2 < par]
        val1 = val1[val2 < par]
    return x,y,val1
verticalEx = 0.1    
ls = LightSource(azdeg=135, altdeg=30)

path_data  = '../data/sar/'

lat_minimum = 19.31
lat_maximum = 19.48
lon_minimum = -155.40
lon_maximum = -155.15

lat_minimum_DEM = 19.31
lat_maximum_DEM = 19.48
lon_minimum_DEM = -155.40
lon_maximum_DEM = -155.15

coherence_thresh = 0.5

step_down = 5 #Donwn sample parameter 
filename = 'S1-GUNW-D-R-087-tops-20180622_20180529-161541-20480N_18471N-PP-652b-v2_0_1.nc'
rootgrp = Dataset(path_data + filename,'r')
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

#Cropping
phase_cropped = phase[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
phase_cropped = phase_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
coh_cropped = coh[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
coh_cropped = coh_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
lat_sar_cropped = lat_sar[(lat_sar > lat_minimum) & (lat_sar < lat_maximum)]
lon_sar_cropped = lon_sar[(lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
phase = phase_cropped[:]
coh = coh_cropped[:]
lat = lat_sar_cropped[:]
lon = lon_sar_cropped[:]

#Downsampling
phase_downsamp = phase[::step_down,::step_down]
coh_downsamp = coh[::step_down,::step_down]
lat_downsamp = lat[::step_down]
lon_downsamp = lon[::step_down]

#Vectorizing
LON,LAT = np.meshgrid(lon,lat)
lon = LON.reshape(1,LON.size)
lat = LAT.reshape(1,LAT.size)
phase = phase.reshape(1,phase.size)
coh = coh.reshape(1,coh.size)
LON,LAT = np.meshgrid(lon_downsamp,lat_downsamp)
lon_downsamp = LON.reshape(1,LON.size)
lat_downsamp = LAT.reshape(1,LAT.size)
phase_downsamp = phase_downsamp.reshape(1,phase_downsamp.size)
coh_downsamp = coh_downsamp.reshape(1,coh_downsamp.size)

#Removing Inchoerent stuff
lon,lat, phase = remove_value(lon,lat,phase,coh,0.5,'gt')
lon_downsamp,lat_downsamp, phase_downsamp = remove_value(lon_downsamp,lat_downsamp,phase_downsamp,coh_downsamp,coherence_thresh,'gt')




rootgrp.close()


'''
DEM Cropping box
'''

'''
GDAL open and coordinates array creation
'''
ds = gdal.Open("../tilt/dem_srtm.tif")
data = ds.ReadAsArray()
gt=ds.GetGeoTransform()
cols = ds.RasterXSize
rows = ds.RasterYSize
minx = gt[0]
miny = gt[3] + cols*gt[4] + rows*gt[5] 
maxx = gt[0] + cols*gt[1] + rows*gt[2]
maxy = gt[3]
lons = np.linspace(minx,maxx,cols)
lats = np.linspace(miny,maxy,rows)
lats = lats[::-1]
#elats = lats[::-1]
dem,lat_DEM,lon_DEM = crop_dem(data,lats,lons,lat_minimum_DEM,lat_maximum_DEM,lon_minimum_DEM,lon_maximum_DEM)    #Cropping the DEM
rgb = ls.shade(dem, cmap=plt.cm.gist_gray, vert_exag=verticalEx, blend_mode='overlay')
'''
Defining parameters for DEM plot
NOTE: as the DEM is plotted with imshow (which is in pixel coordinates), the llclon... parameters 
must correspond to the corner of the matrixm otherwise georeferencing and subsequent coordinate plot DO NOT WORK!!! 
'''
llclon = np.min(lon_DEM)
llclat = np.min(lat_DEM)
urclon = np.max(lon_DEM)
urclat = np.max(lat_DEM)
fig1 = plt.figure(1,figsize=(8,4))
map1 = Basemap(projection='tmerc', 
              lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
              llcrnrlon=llclon, 
              llcrnrlat=llclat, 
              urcrnrlon=urclon, 
              urcrnrlat=urclat)
lon_map,lat_map = map1(lon,lat)
im1 = map1.imshow(rgb,origin = 'upper')
im2 = map1.scatter(lon_map,lat_map,c = phase,alpha = 0.01 ,cmap = 'seismic',vmin = -30,vmax =30)

plt.title('Original')

fig2 = plt.figure(2,figsize=(8,4))
map2 = Basemap(projection='tmerc', 
              lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
              llcrnrlon=llclon, 
              llcrnrlat=llclat, 
              urcrnrlon=urclon, 
              urcrnrlat=urclat)
lon_downsamp_map,lat_downsamp_map = map1(lon_downsamp,lat_downsamp)
im1 = map2.imshow(rgb,origin = 'upper',rasterized =True)
im2 = map2.scatter(lon_downsamp_map,lat_downsamp_map,s = 10,c = phase_downsamp,alpha = 0.6 ,cmap = 'seismic',vmin = -30,vmax =30)
plt.title('Downsampled with a step of '+ str(step_down))
interferogram = {}
interferogram['phase'] = phase_downsamp
interferogram['lat'] = lat_downsamp
interferogram['lon'] = lon_downsamp
interferogram['inc'] = incidence
interferogram['azi'] = azimuth
interferogram['look'] = look
interferogram ['latMeta'] = latMeta
interferogram ['lonMeta'] = lonMeta
interferogram ['lonMeta'] = lonMeta
interferogram ['heightsMeta'] = heightsMeta 
interferogram['wavelength'] = wavelength
plt.savefig( 'Figs/test.pdf',dpi = 600)
pickle.dump(interferogram,open(path_data + filename[:3] + filename[21:38] + '_coh_0.5.pickle','wb'))

