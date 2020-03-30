#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:19:25 2019

@author: aroman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a figure for each tilt station, a master figure showing different stations together, and a figure for one chosen station which allows to analyze the stick-slip cycles.
The figure of each station is a subplot containing 
1) The North component as function of the East component
2) The North comp. and and East component of function of time
3) The first and second component extracted by PCA as function of time
The master figure contains in each subplot the first component of PCA for each station in the list as function of time. One extra subplot contains the time between collapses event as function of time (calculated from last station in the list).
The last figure contains the results of the peak analysis 

Created on Wed Sep 25 16:35:48 2019

@author: aroman
"""
def get_data_segment(date_start,date_end,dates):
    #extract data between to dates expresses in human format
    number_start = mdates.date2num(datetime.strptime(date_start,'%Y-%m-%d'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%Y-%m-%d'))
    ind = (dates>number_start) & (dates<number_end)
    return ind,number_start,number_end


import pickle
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import signal
from sklearn.linear_model import LinearRegression
from PCA_projection_tilt import *


plt.rcParams['svg.fonttype'] = 'none'

stations = pickle.load(open('tilt_dictionary_01may.pickle','rb')) 
'''
Find mininimum and maximum time for colormap
'''
tmin = 1e+7
tmax = 0
#b, a = signal.butter(2, 0.03)
blow, alow = signal.butter(2, 1e-4)

months = mdates.MonthLocator()  # every month
fifteen_days = mdates.DayLocator(bymonthday=(1,15))  # labels every 15 day
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')
list_station = ['UWD']  #Select station to plot

date_eruption = mdates.date2num(datetime.strptime('2018-05-03','%Y-%m-%d'))
date_eruption_end = mdates.date2num(datetime.strptime('2018-08-04','%Y-%m-%d'))
date_caldera_start = mdates.date2num(datetime.strptime('2018-05-26','%Y-%m-%d'))
date_caldera_end = mdates.date2num(datetime.strptime('2018-08-03','%Y-%m-%d'))
#day_beginning = '2018-07-10' #Few cycles
#day_end = '2018-07-20'
day_beginning = '2018-05-01' #all timeseries
day_end = '2018-08-08'
first_day = mdates.date2num(datetime.strptime(day_beginning,'%Y-%m-%d'))
last_day = mdates.date2num(datetime.strptime(day_end,'%Y-%m-%d'))

for name in list_station:
    east = stations[name]['east']
    north = stations[name]['north']
    time = stations[name]['time']
    tmax = max(tmax,np.max(time))
    tmin = min(tmin,np.min(time))
    

name ='IKI'

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
north_low = signal.filtfilt(blow, alow,north)
east_low = signal.filtfilt(blow, alow,east)

plt.plot(time,east)
plt.plot(time,east_low)
bhigh, ahigh = signal.butter(2, 1e-4,'high')
north_high = signal.filtfilt(bhigh, ahigh,north)
east_high = signal.filtfilt(bhigh, ahigh,east)
b, a = signal.butter(2, 0.03)
north_high = signal.filtfilt(b, a,north_high)
east_high = signal.filtfilt(b, a,east_high)

plt.plot(time,east_high)