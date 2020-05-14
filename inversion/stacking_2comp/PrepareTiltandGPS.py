#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:53:57 2019

@author: aroman
"""
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import csv 
import pickle
from PCA_projection_tilt import *

def format_gps(A):
    A = A[5:]
    t = []
    d = []
    uz = []
    for i in A:
        t.append(float(i[0]))
        d.append(i[1])
        uz.append(float(i[2])) 
    t = np.array(t)
    uz = np.array(uz)
    return t,d,uz

def get_data_segment(date_start,date_end,dates):
    #extract data between to dates expresses in human format
    number_start = mdates.date2num(datetime.strptime(date_start,'%Y-%m-%d'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%Y-%m-%d'))
    ind = (dates>number_start) & (dates<number_end)
    return ind,number_start,number_end
def read_csv(filename):
  
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        A = []
        for row in readCSV:
            A.append(row)
    return A


def stick_slip(tilt_filt,dates,cald_start,station,fmt,majloc,minloc):
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
    else:
        amp = 0
    print('Mean inflation event of station ', station,' is ', np.mean(amp))
    return np.diff(time_min_peaks),time_min_peaks[:-1],peaks,min_peaks

name = 'UWD' #Tilt station to invert
path_data = '../../../data/'
stations = pickle.load(open(path_data + 'tilt_dictionary_01may.pickle','rb')) 
path_GPS = '../../GPS/'
b, a = signal.butter(2, 0.03)
date_caldera_start = mdates.date2num(datetime.strptime('2018-05-26','%Y-%m-%d'))
date_caldera_end = mdates.date2num(datetime.strptime('2018-08-03','%Y-%m-%d'))
#tmin = 1e+7
#tmax = 0
months = mdates.MonthLocator()  # every month
fifteen_days = mdates.DayLocator(bymonthday=(1,15))  # labels every 15 day
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')

day_beginning = '2018-05-01' #all timeseries
day_end = '2018-08-08'
first_day = mdates.date2num(datetime.strptime(day_beginning,'%Y-%m-%d'))
last_day = mdates.date2num(datetime.strptime(day_end,'%Y-%m-%d'))

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
proj_max = east * u1 + north * w1
proj_min = east * u2 + north * w2
proj_max = signal.filtfilt(b, a, proj_max)
dt_stick,time_collapses,tilt_stick,tilt_slip = stick_slip(proj_max,time,date_caldera_start,name,date_fmt,months,days)
nCollapses = np.arange(1,len(dt_stick)+1)

gps_list = ['CALS.csv']
for gps in gps_list:
    data = read_csv(path_GPS + gps)
    time_gps,date,disp = format_gps(data)
    date_new = []
    for line in date:
        date_temp = line[:10]
        date_new.append(mdates.date2num(datetime.strptime(date_temp,'%Y-%m-%d')))
    date_gps = np.array(date_new)
nCollapses = nCollapses[:len(time_collapses)]
tilt_slip = tilt_slip[:len(time_collapses)]
tilt_stick = tilt_stick[:len(time_collapses)]
dt_stick = dt_stick[:len(time_collapses)]
tilt_stick = tilt_stick[time_collapses>= date_gps[0]]
tilt_slip = tilt_slip[time_collapses>= date_gps[0]]
dt_stick = dt_stick[time_collapses>= date_gps[0]]
nCollapses =nCollapses[time_collapses>= date_gps[0]]
time_collapses = time_collapses[time_collapses>= date_gps[0]]

f = interp1d(date_gps,disp)
disp_interp = f(time_collapses)
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8,10))
ax[0].plot(nCollapses,tilt_slip,'bo')
ax[0].plot(nCollapses,tilt_stick,'ro')
ax[1].plot(nCollapses,disp_interp,'go')
plt.show()
N = input('From which collapse you want to proceed with inversion?')
N = int(N)
tilt_stick = tilt_stick[nCollapses >= N]
tilt_slip = tilt_slip[nCollapses >= N]
dt_stick = dt_stick[nCollapses >= N]
gps = disp_interp[nCollapses >= N]
nCollapses = nCollapses[nCollapses >= N]
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8,10))
ax[0].plot(nCollapses,tilt_slip,'bo')
ax[0].plot(nCollapses,tilt_stick,'ro')
ax[1].plot(nCollapses,gps,'go')
data = {}
data['tilt_stick'] = tilt_stick
data['tilt_slip'] = tilt_slip
data['dt_stick'] = dt_stick
data['gps'] = gps - gps[0]
data['tilt_std'] = np.std(proj_min)
data['LonLat'] = np.array([[stations[name]['lon_lat'][0]],[stations[name]['lon_lat'][1]]])
pickle.dump(data,open('DataForInversion' + name + '.pickle','wb'))








