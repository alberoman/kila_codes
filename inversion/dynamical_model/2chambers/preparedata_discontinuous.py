#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a figure for each tilt station, a master figure showing different vol together, and a figure for one chosen station which allows to analyze the stick-slip cycles.
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

def stick_slip(tilt_filt,dates,cald_start,station,fmt,majloc,minloc,beginning,end):
    if station == 'SDH':
        dist = 400
        prom = 7
    else:
        dist = 500
        prom = 10
        
    ind, _ = signal.find_peaks(tilt_filt,distance = dist,prominence= prom)
    time_peaks = dates[ind]
    peaks = tilt_filt[ind]
    min_peaks = []
    time_min_peaks = []
    for i in range(len(ind)-1):
        y = tilt_filt[ind[i]:ind[i+1]]
        x = dates[ind[i]:ind[i+1]]
        minimum = np.min(y)
        min_peaks.append(minimum)
        time_min_peaks.append(x[np.where(y==minimum)][0])
    print(ind)
    if len(ind)>0:
        y = tilt_filt[ind[-1]:]
        x = dates[ind[-1]:]
        minimum = np.min(y)
        min_peaks.append(minimum)
        time_min_peaks.append(x[np.where(y==minimum)][0])
        peaks = np.array(peaks)
        min_peaks =np.array(min_peaks[:-1])
        time_min_peaks = np.array(time_min_peaks[:-1])
        time_peaks = np.array(time_peaks)
        min_peaks = min_peaks[time_min_peaks > cald_start]
        time_min_peaks = time_min_peaks[time_min_peaks > cald_start]
        peaks = peaks[time_peaks > cald_start]
        time_peaks = time_peaks[time_peaks > cald_start]
        amp = peaks[1:] - min_peaks
        print('Number of collapses: ',len(time_peaks))
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)
        ax.plot(dates,tilt_filt)
        ax.plot(time_peaks,peaks,'ro-')
        ax.plot(time_min_peaks,min_peaks,'bo-')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(majloc)
        ax.xaxis.set_minor_locator(minloc)
        ax.set_xlim([beginning,end])
        ax = fig.add_subplot(3,1,2)
        ax.plot(time_min_peaks[:-1],np.diff(time_min_peaks))
        ax.plot(time_min_peaks[:-1],np.diff(time_min_peaks),'o')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(majloc)
        ax.xaxis.set_minor_locator(minloc)
        ax.set_xlim([beginning,end])
        ax = fig.add_subplot(3,1,3)
        ax.plot(time_min_peaks,amp,'o')
        ax.plot(time_min_peaks,amp)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(majloc)
        ax.xaxis.set_minor_locator(minloc)
        ax.set_xlim([beginning,end])
    else:
        amp = 0
    print('Mean inflation event of station ', station,' is ', np.mean(amp))

    return np.diff(time_min_peaks),time_min_peaks[:-1],amp

import pickle
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import signal



stations = pickle.load(open('../../../tilt/tilt_dictionary_01may_discontinuous.pickle','rb')) 
'''
Find mininimum and maximum time for colormap
'''
tmin = 1e+7
tmax = 0
b, a = signal.butter(2, 0.03)
months = mdates.MonthLocator()  # every month
fifteen_days = mdates.DayLocator(bymonthday=(1,15))  # labels every 15 day
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')
list_station = ['UWD','SDH','IKI',]  #Select station to plot

date_eruption = mdates.date2num(datetime.strptime('2018-05-03','%Y-%m-%d'))
date_eruption_end = mdates.date2num(datetime.strptime('2018-08-04','%Y-%m-%d'))
date_caldera_start = mdates.date2num(datetime.strptime('2018-05-26','%Y-%m-%d'))
date_caldera_end = mdates.date2num(datetime.strptime('2018-08-03','%Y-%m-%d'))
#day_beginning = '2018-07-10' #Few cycles
#day_end = '2018-07-20'
day_beginning = '2018-05-01' #all timeseries
day_end = '2018-08-06'
first_day = mdates.date2num(datetime.strptime(day_beginning,'%Y-%m-%d'))
last_day = mdates.date2num(datetime.strptime(day_end,'%Y-%m-%d'))



'''
Start plotting
'''
for name in list_station:
    fig = plt.figure()
    east = stations[name]['east']
    north = stations[name]['north']
    time = stations[name]['time']
    '''
    Extract staff that is not nan
    '''

    east = east[::1]
    for i in range(len(east)):
        axx = fig.add_subplot(211)
        axx.plot(time[i],north[i])
        axy = fig.add_subplot(212)
        axy.plot(time[i],east[i])

plt.show()

