#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot DEM of Kilauea and plot directions of tilts which shows the maximum variance over the time period. The direction of maximum variance is extracted by taking the first component of PCA. Use tilt data that has been formatted by format_tilt.py
Created on Wed Sep 25 16:35:48 2019

@author: aroman
"""
def get_data_segment(date_start,date_end,dates):
    #extract data between to dates expresses in human format
    number_start = mdates.date2num(datetime.strptime(date_start,'%m/%d/%Y'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%m/%d/%Y'))
    ind = (dates>number_start) & (dates<number_end)
    return ind,number_start,number_end

import pickle
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import gdal
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from PCA_projection_tilt import *

plt.rcParams['svg.fonttype'] = 'none'

stations = pickle.load(open('tilt_dictionary_01may.pickle','rb')) 
list_station =  ['UWD','SDH','JKA','IKI','POO']
tmax = 0

N_station = len(list_station)
'''
DEM Cropping box
'''
lat_minimum = 19.30 
lat_maximum = 19.46
lon_minimum = -155.31
lon_maximum = -154.90
'''
GDAL open and coordinates array creation
'''
ds = gdal.Open("dem_srtm.tif")
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
llclon = np.min(lons)
llclat = np.min(lats)
urclon = np.max(lons)
urclat = np.max(lats)
[X,Y] = np.meshgrid(lons,lats)

'''
Plot the DEM in Figure 1
'''
fig1 = plt.figure(1,figsize=(8,4))
map = Basemap(projection='tmerc', 
              lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
              llcrnrlon=llclon, 
              llcrnrlat=llclat, 
              urcrnrlon=urclon, 
              urcrnrlat=urclat)

im1 = map.imshow(shadow,origin = 'upper',cmap = 'Greys')
im2 = map.imshow(dem,origin = 'upper',cmap = 'terrain',alpha =4.3,clim = (-3000,5000) )
'''
Start plotting tiltmeters position and tilt on stereonets
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
    draw_vector([x+200,y+200],[x+200 +u1*750,y+200 + w1*750 ],'orange')
    draw_vector([x+200,y+200],[x+200 -u1*750,y+200 - w1*750 ],'orange')
    if not (name == 'UWE'):             #UWD and UWE are on the same site, plotting only UWE
        map.plot(x,y,'ko',markersize = 4)
        plt.text(x+500,y+150,name,color = 'black',fontsize = 14,fontweight = 'heavy')
    
    counter = counter + 1
    plt.savefig('Figs/dem_allStations_01may.pdf',dpi = 300)
plt.show()
