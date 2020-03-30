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

lat_minimum = 19.3 
lat_maximum = 19.46
lon_minimum = -155.40
lon_maximum = -154.9
filename = 'S1-GUNW-D-R-087-tops-20180622_20180529-161541-20480N_18471N-PP-652b-v2_0_1.nc'
rootgrp = Dataset(filename,'r')
data= rootgrp['science/grids/data/']  
lon_sar = data.variables['longitude'][:]
lat_sar = data.variables['latitude'][:]
phase = data.variables['unwrappedPhase'][:]
coh = data.variables['coherence'][:]
rootgrp.close()
phase_cropped = phase[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
phase_cropped = phase_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
coh_cropped = coh[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
coh_cropped = coh_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
lat_sar_cropped = lat_sar[(lat_sar > lat_minimum) & (lat_sar < lat_maximum)]
lon_sar_cropped = lon_sar[(lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
phase = phase_cropped[:]
coh = coh_cropped[:]

lat_sar = lat_sar_cropped[:]
lon_sar = lon_sar_cropped[:]
lat_sar = lat_sar[::-1]
list_station =  ['UWD','SDH','IKI']
stations = pickle.load(open('../tilt/tilt_dictionary_20june.pickle','rb')) 
N_station = len(list_station)

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
dem,lats,lons = crop_dem(data,lats,lons,lat_minimum,lat_maximum,lon_minimum,lon_maximum)    #Cropping the DEM
shadow = hillshade(dem,30,30)   #Creating shaded relief
'''
Defining parameters for DEM plot
NOTE: as the DEM is plotted with imshow (which is in pixel coordinates), the llclon... parameters 
must correspond to the corner of the matrixm otherwise georeferencing and subsequent coordinate plot DO NOT WORK!!! 
'''
dem[dem <= 5] =-5000
f_phase = interpolate.interp2d(lon_sar,lat_sar,phase)
f_coh = interpolate.interp2d(lon_sar,lat_sar,coh)
phase_interp = f_phase(lons,lats)
coh_interp = f_coh(lons,lats)
phase_interp[coh_interp<0.2] = np.nan

llclon = np.min(lons)
llclat = np.min(lats)
urclon = np.max(lons)
urclat = np.max(lats)
[X,Y] = np.meshgrid(lons,lats)
fig1 = plt.figure(1,figsize=(8,4))

map = Basemap(projection='tmerc', 
              lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
              llcrnrlon=llclon, 
              llcrnrlat=llclat, 
              urcrnrlon=urclon, 
              urcrnrlat=urclat)

im1 = map.imshow(shadow,origin = 'upper',cmap = 'Greys')
im2 = map.imshow(phase_interp,origin = 'upper',cmap = 'seismic',alpha =0.6,clim = [-30,30])
plt.colorbar(im2)
'''
Start plotting tiltmeters position and tilt 
'''
counter = 0
for name in list_station:
    x,y = map(stations[name]['lon_lat'][0],stations[name]['lon_lat'][1])

    east = stations[name]['east']
    north = stations[name]['north']
    time = stations[name]['time']
    '''
    Extract staff that is not nan
    '''
    npix =  750 # extension (in pixel) of the arrows
    indices = (np.isnan(north) == False) & (np.isnan(east) == False)
    time = time[indices] 
    east = east[indices] 
    north = north[indices] 
    east = (east - east[0]) 
    north = (north - north[0])
    time = time[::1]
    north = north[::1]
    east = east[::1]
    u1,w1,u2,w2,PCAmean = PCA_vectors(east,north) #Extract PCA vector and PCA mean
    plt.figure(1)
    draw_vector([x,y],[x +u1*npix,y + w1*npix ],'orange')
    draw_vector([x,y],[x -u1*npix,y - w1*npix ],'orange')
    if not (name == 'UWE'):             #UWD and UWE are on the same site, plotting only UWE
        map.plot(x,y,'ko',markersize = 8)
    if (name == 'SDH'):
        plt.text(x-1000,y+800,name,color = 'black',fontsize = 10,fontweight = 'heavy')
    else:
        plt.text(x+500,y+150,name,color = 'black',fontsize = 10,fontweight = 'heavy')
    
    counter = counter + 1
    
'''
Plotting GPS
'''

myProj = Proj("+proj=utm +zone=5, +north +ellps=GRS80 +datum=WGS84 +units=m +no_defs")
lon_ref = -155.3
mod_max = 0
position = {}
position['CRIM'] = [-155.274,19.395]
position['OUTL'] = [-155.281,19.387]
position['CNPK'] = [-155.306,19.392]
position['BYRL'] = [-155.260,19.412]
position['UWEV'] = [-155.291,19.421]
position['KOSM'] = [-155.316,19.363]
position['AHUP'] = [-155.266,19.379]
GPS = pickle.load(open('../GPS/nevada/dict_GPS.pickle','rb'))
#for station in GPS.keys():
#    north = GPS[station]['north'][-1] - GPS[station]['north'][0]
#    east = GPS[station]['east'][-1] - GPS[station]['east'][0]
#    mod = (north**2 + east**2)**0.5
#    mod_max = max(mod,mod_max)
#constant = 2 * npix / mod_max
#for station in GPS.keys():
#    north = GPS[station]['north'][-1] - GPS[station]['north'][0]
#    east = GPS[station]['east'][-1] - GPS[station]['east'][0]
#    mod = (north**2 + east**2)**0.5
#    x,y = map(position[station][0],position[station][1])
#    ratio = north / east 
#    draw_vector([x,y],[x + east * constant,y + north * constant ],'black',1)



plt.show()

