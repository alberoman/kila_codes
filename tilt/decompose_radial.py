#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:27:39 2020

@author: aroman
"""

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
golden_ratio =  1.618
onepltsize = (golden_ratio*2.5,2.5)
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



list_station =  ['UWD',]
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

    lonTemp = stations[name]['lon_lat'][0] * np.ones(len(east))   
    latTemp = stations[name]['lon_lat'][1] * np.ones(len(east))
    LonLat= np.concatenate((lonTemp.reshape(1,lonTemp.size),latTemp.reshape(1,latTemp.size)),0)
    xy = llh2local(LonLat, ref_coord)
    xy = xy.transpose() * 1000
    
xsrc1 = results['MAP']['xSource1']
ysrc1 = results['MAP']['ySource1']
xsrc2 = results['MAP']['xSource2']
ysrc2 = results['MAP']['ySource2']
xst = xy[0][0]
yst = xy[0][1]
xrad1= xst - xsrc1
yrad1= yst - ysrc1
xrad1 = xrad1/(xrad1**2 + yrad1**2)**0.5
yrad1 = yrad1/(xrad1**2 + yrad1**2)**0.5


xrad2= xst - xsrc2
yrad2= yst - ysrc2
xrad2 = xrad2/(xrad2**2 + yrad2**2)**0.5
yrad2 = yrad2/(xrad2**2 + yrad2**2)**0.5
y = stations['SDH']['north']
x = stations['SDH']['east']
time = stations
proj1 = x* xrad1 + y * yrad1
proj2 = x* xrad2 + y * yrad2
plt.figure(figsize = onepltsize)
plt.plot(time_ref,proj1)
plt.plot(time_ref,proj2)

ax = plt.gca()
ax.set_xlim(date_initial_plot,date_final_plot)
ax.xaxis.set_major_formatter(date_fmt)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
ax.set_ylabel('SDH [microrad]',fontsize = 14)
ax.set_xlabel('Date',fontsize = 14)
ax.tick_params(labelsize = 12)
plt.savefig('tilt_decomposition.pdf')

        
        
        

        