#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:00:52 2019

@author: aroman
"""

import numpy as np
import gdal
from sklearn.decomposition import PCA
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pickle
from scipy.interpolate import interp1d 
from matplotlib import animation
from scipy import signal
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
import sys
sys.path.insert(1,'../sar/')
from coord_conversion import *

date_initial_str = '2018-07-01'
date_final_str = '2018-08-05'
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')
date_initial_plot = mdates.date2num(datetime.strptime('2018-07-01','%Y-%m-%d'))
date_initial_volumes = mdates.date2num(datetime.strptime('2018-05-01','%Y-%m-%d')) #for GLISTEN volume
date_final_plot = mdates.date2num(datetime.strptime('2018-08-05','%Y-%m-%d'))
date_eruption = mdates.date2num(datetime.strptime('2018-05-04','%Y-%m-%d'))
date_eruption_end = mdates.date2num(datetime.strptime('2018-08-04','%Y-%m-%d'))
date_explosion1 = mdates.date2num(datetime.strptime('2018-05-16','%Y-%m-%d'))
date_explosion2 = mdates.date2num(datetime.strptime('2018-05-26','%Y-%m-%d'))
date_intensity_eruption = mdates.date2num(datetime.strptime('2018-05-18','%Y-%m-%d'))
date_caldera_start = mdates.date2num(datetime.strptime('2018-05-26','%Y-%m-%d'))
date_caldera_end = mdates.date2num(datetime.strptime('2018-08-02','%Y-%m-%d'))
date_caldera_broad = mdates.date2num(datetime.strptime('2018-06-20','%Y-%m-%d'))

def crop_dem(array,lat_orig,lon_orig,lat_min,lat_max,lon_min,lon_max):
    array = array[(lat_orig >= lat_min) & (lat_orig <= lat_max),:]
    array = array[:,(lon_orig >= lon_min) & (lon_orig <= lon_max)]
    return array, lat_orig[(lat_orig >= lat_min) & (lat_orig <= lat_max)],lon_orig[(lon_orig >= lon_min) & (lon_orig <= lon_max)]

def draw_vector(v0, v1,cl, ax ,head =0):
    #Draws a vector with no arrow using annotation. v0 and v1 are two vectors containing the coordinates of the starting point of the arrow (v0)
    #and the final point of the arrow (v1)
    if head:
        arrowprops=dict(arrowstyle = '->, head_width=0.3, head_length=0.5',color= cl,
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    else:
        arrowprops=dict(arrowstyle='-',color= cl,
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    h = ax.annotate('', v1, v0, arrowprops=arrowprops)
    return(h)
    


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

path_data_sar  = '../data/sar/'
lat_minimum = 19.31
lat_maximum = 19.48
lon_minimum = -155.40
lon_maximum = -155.15

lat_minimum_DEM = 19.31
lat_maximum_DEM = 19.48
lon_minimum_DEM = -155.40
lon_maximum_DEM = -155.15

list_station =  ['UWD','SDH','IKI']
filename = '../../data/Mogi_Metropolis_100000_2mogi.pickle'
results =pickle.load(open(filename,'rb'))
ref_coord = np.array([-155.274,19.395]) #Cordinate of CRIM station

coord_source1 = np.zeros((2,1))
coord_source1[0,0] = results['MAP']['xSource1'] / 1000
coord_source1[1,0] = results['MAP']['ySource1'] / 1000
coord_source2 = np.zeros((2,1))
coord_source2[0,0] = results['MAP']['xSource2'] / 1000
coord_source2[1,0] = results['MAP']['ySource2'] / 1000

lonlatSource1 = local2llh(coord_source1,ref_coord)
lonlatSource2 = local2llh(coord_source2,ref_coord)

#
#filename = 'S1-GUNW-D-R-087-tops-20180622_20180529-161541-20480N_18471N-PP-652b-v2_0_1.nc'
#rootgrp = Dataset(path_data_sar + filename,'r')
#data= rootgrp['science/grids/data/']
#geometry = rootgrp['science/grids/imagingGeometry/']
#wavelength = np.float64(rootgrp['science/radarMetaData'].variables['wavelength'][:])
#lonMeta = geometry.variables['longitudeMeta'][:]
#latMeta = geometry.variables['latitudeMeta'][:]
#heightsMeta = geometry.variables['heightsMeta'][:]
#heightsMeta = heightsMeta[:]
#incidence = geometry.variables['incidenceAngle'][:]
#look = geometry.variables['lookAngle'][:]
#azimuth = geometry.variables['azimuthAngle'][:]
#lon_sar = data.variables['longitude'][:]
#lat_sar = data.variables['latitude'][:]
#phase = data.variables['unwrappedPhase'][:]
#coh = data.variables['coherence'][:]
#
##Cropping
#phase_cropped = phase[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
#phase_cropped = phase_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
#coh_cropped = coh[(lat_sar > lat_minimum) & (lat_sar < lat_maximum),:]
#coh_cropped = coh_cropped[:, (lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
#lat_sar_cropped = lat_sar[(lat_sar > lat_minimum) & (lat_sar < lat_maximum)]
#lon_sar_cropped = lon_sar[(lon_sar > lon_minimum) & (lon_sar < lon_maximum)]
#phase = phase_cropped[:]
#coh = coh_cropped[:]
#lat = lat_sar_cropped[:]
#lon = lon_sar_cropped[:]
#
##Downsampling
#phase_downsamp = phase[::step_down,::step_down]
#coh_downsamp = coh[::step_down,::step_down]
#lat_downsamp = lat[::step_down]
#lon_downsamp = lon[::step_down]
#
##Vectorizing
#LON,LAT = np.meshgrid(lon,lat)
#lon = LON.reshape(1,LON.size)
#lat = LAT.reshape(1,LAT.size)
#phase = phase.reshape(1,phase.size)
#coh = coh.reshape(1,coh.size)
#LON,LAT = np.meshgrid(lon_downsamp,lat_downsamp)
#lon_downsamp = LON.reshape(1,LON.size)
#lat_downsamp = LAT.reshape(1,LAT.size)
#phase_downsamp = phase_downsamp.reshape(1,phase_downsamp.size)
#coh_downsamp = coh_downsamp.reshape(1,coh_downsamp.size)
##Removing Inchoerent stuff
#lon,lat, phase = remove_value(lon,lat,phase,coh,0.5,'gt')
#lon_downsamp,lat_downsamp, phase_downsamp = remove_value(lon_downsamp,lat_downsamp,phase_downsamp,coh_downsamp,coherence_thresh,'gt')
#los_downsamp = phase_downsamp /(4 * 3.14) * wavelength
#
#


#rootgrp.close()


'''
DEM Cropping box
'''

'''
GDAL open and coordinates array creation
'''
ds = gdal.Open("../data/dem_srtm.tif")
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
#ax2 = fig2.add_subplot(111)
#im_dummy = ax2.scatter(lon_downsamp_map,lat_downsamp_map,s = 20,c = los_downsamp*100,alpha = 1.0 ,cmap = 'Blues_r',vmin = -16, vmax =0)

list_station =  ['UWD','SDH','IKI']
stations = pickle.load(open('../../data/tilt_dictionary_01july.pickle','rb')) 
time_tilt = stations['UWD']['time']
east = stations['UWD']['east']
north = stations['UWD']['north']
indices = (np.isnan(north) == False) & (np.isnan(east) == False)
time_tilt = time_tilt[indices]


b, a = signal.butter(2, 0.03)
for name in list_station:
    time = stations[name]['time']
    east = stations[name]['east']
    east = signal.filtfilt(b, a, east)
    north = stations[name]['north']
    north = signal.filtfilt(b, a, north)

    indices_in = np.logical_and(time_tilt >= np.min(time),time_tilt <= np.max(time))
    indices_out = np.logical_or(time_tilt < np.min(time),time_tilt > np.max(time))
    time_ref = time_tilt[indices_in]
    dummy_vect_east =np.ones(len(time_tilt))
    dummy_vect_north =np.ones(len(time_tilt))
    first_ind = np.where(indices_in)[0][0]
    f =  interp1d(time,east)
    dummy_vect_east[indices_in] = f(time_ref)
    dummy_vect_east[indices_out] = np.nan
    stations[name]['east'] = dummy_vect_east - dummy_vect_east[first_ind]
    
    f =  interp1d(time,north)
    dummy_vect_north[indices_in] = f(time_ref)
    dummy_vect_north[indices_out] = np.nan
    stations[name]['north'] = dummy_vect_north - dummy_vect_north[first_ind]

fig = plt.figure(figsize =(8.,8.0))
gs = fig.add_gridspec(11, 10)
ax_map = fig.add_subplot(gs[:5, :])
ax_UWD = fig.add_subplot(gs[5:8,2:-2])
ax_SDH = fig.add_subplot(gs[8:,2:-2])

map = Basemap(projection='tmerc', 
              lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
              llcrnrlon=llclon, 
              llcrnrlat=llclat, 
              urcrnrlon=urclon, 
              urcrnrlat=urclat,ax = ax_map)
 
#lon_downsamp_map,lat_downsamp_map = map(lon_downsamp,lat_downsamp)
im1 = map.imshow(rgb,origin = 'upper',rasterized =True)
#im2 = map.scatter(lon_downsamp_map,lat_downsamp_map,s = 20,c = los_downsamp*100,alpha = 0.6 ,cmap = 'Blues_r',vmin = -16, vmax =0)
#z$cbar = map.colorbar(im2,shrink = 0.2,alpha = 1,ticks=[ 0,-7.5,-15])
#cbar.ax.set_yticklabels(['0', '-7.5','-15']) 
#cbar.set_label('LOS displacements [cm]',fontsize = 18)
#cbar.solids.set_rasterized(True)
npix =  750 # extension (in pixel) of the arrows
magnitude1 = (stations['UWD']['north'])
magnitude2 = (stations['SDH']['north'])

filename_counter = 0
step = 300
for counter in np.arange(0,len(time_tilt),step):
    h1 = []
    h2 = []
    for name in list_station:
        x,y = map(stations[name]['lon_lat'][0],stations[name]['lon_lat'][1])
        east = stations[name]['east'][counter]
        north = stations[name]['north'][counter]
        '''
        Extract staff that is not nan
        '''
        east = east /(east**2 + north**2)**0.5
        north = north / (east**2 + north**2)**0.5
        handle1 = draw_vector([x,y],[x + east * npix,y + north*npix ],'orange', ax_map)
        handle2 = draw_vector([x,y],[x -east*npix,y - north*npix ],'orange', ax_map)
        if not (name == 'UWE'):             #UWD and UWE are on the same site, plotting only UWE
            map.plot(x,y,'ko',markersize = 4)
            ax_map.text(x-2000,y-1000,name,color = 'black',fontsize = 7,fontweight = 'heavy')
        h1.append(handle1)
        h2.append(handle2)
    print(counter)
    timeseries1 = ax_UWD.plot(time_tilt[::step],magnitude1[::step],'blue')
    timeseries2 = ax_SDH.plot(time_tilt[::step],magnitude2[::step],'blue')
    time_segment = time_tilt[:(filename_counter * step):step]
    magnitude_segment1 = magnitude1[:(filename_counter * step):step]
    magnitude_segment2 = magnitude2[:(filename_counter * step):step]
    ax_UWD.plot(time_segment,magnitude_segment1,'r')
    ax_SDH.plot(time_segment,magnitude_segment2,'r')

    ax_UWD.set_xlim(date_initial_plot,date_final_plot)
    ax_UWD.xaxis.set_major_formatter(date_fmt)
    ax_UWD.xaxis.set_major_locator(months)
    ax_UWD.xaxis.set_minor_locator(days)
    ax_SDH.set_xlim(date_initial_plot,date_final_plot)
    ax_SDH.xaxis.set_major_formatter(date_fmt)
    ax_SDH.xaxis.set_major_locator(months)
    ax_SDH.xaxis.set_minor_locator(days)
    ax_UWD.set_ylabel('UWD N-S [microrad]')
    ax_SDH.set_ylabel('SDH N-S [microrad]')

    if len(str(filename_counter)) == 1:
        filename = '000' + str(filename_counter)
    elif len(str(filename_counter)) == 2:
        filename = '00' + str(filename_counter)
    elif len(str(filename_counter)) == 3:
        filename = '0' + str(filename_counter)
    elif len(str(filename_counter)) == 4:
        filename = str(filename_counter)
    plt.savefig('../../results/movie_tilt/' + str(filename) + '.png',dpi = 300)
    filename_counter = filename_counter + 1 
    h1[0].remove()
    h1[1].remove()
    h1[2].remove()
    h2[0].remove()
    h2[1].remove()
    h2[2].remove()
    timeseries1[0].remove()
    timeseries2[0].remove()


        
        
        

        