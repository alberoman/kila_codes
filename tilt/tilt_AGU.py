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
        amp = peaks - min_peaks
        print('Number of collapses: ',len(time_peaks))
        fig = plt.figure(figsize =(8,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(dates,tilt_filt)
        ax.plot(time_peaks,peaks,'bo-')
        ax.plot(time_min_peaks,min_peaks,'ro-')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(majloc)
        ax.xaxis.set_minor_locator(minloc)
        ax.set_xlim([beginning,end])
        ax.set_ylabel('Tilt [$\mu$rad]',fontsize = 24)
        plt.savefig('tilt_agu.svg')
        amp = 0
    print('Mean inflation event of station ', station,' is ', np.mean(amp))

    return np.diff(time_min_peaks),time_min_peaks[:-1],peaks,min_peaks

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
import matplotlib as mpl
mpl.style.use('dark_background')


plt.rcParams['svg.fonttype'] = 'none'

stations = pickle.load(open('tilt_dictionary_01may.pickle','rb')) 
'''
Find mininimum and maximum time for colormap
'''
tmin = 1e+7
tmax = 0
#b, a = signal.butter(2, 0.03)
b, a = signal.butter(2, 1e-1)

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
day_beginning = '2018-05-20' #all timeseries
day_end = '2018-08-08'
first_day = mdates.date2num(datetime.strptime(day_beginning,'%Y-%m-%d'))
last_day = mdates.date2num(datetime.strptime(day_end,'%Y-%m-%d'))

for name in list_station:
    east = stations[name]['east']
    north = stations[name]['north']
    time = stations[name]['time']
    tmax = max(tmax,np.max(time))
    tmin = min(tmin,np.min(time))
    
N_station = 1

#h_station = 5 #for few cycles
#w_station = 5 #for few cycles
h_station = 4 #For the all timeseries
w_station = 10    #for the all time series
h_total_station = h_station * (N_station + 1) # if you want only stations and not period

h_figure=  h_total_station 

fig_main = plt.figure(1,figsize = (w_station,h_figure))

'''
Start ploitting
'''

counter = 0
name ='UWD'

fig = plt.figure()
gs = fig.add_gridspec(6, 6)

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
ax1 = fig.add_subplot(gs[:, :3])    
cm = ax1.scatter(east,north,c = time, vmin = tmin, vmax = tmax,alpha = 0.6)
ax1.quiver(PCAmean[0],PCAmean[1],u1,w1,angles='xy', scale_units='xy', scale = 0.1)
ax2 = fig.add_subplot(gs[:3, 4:]) 
ax2.plot(time,east,'g')
ax2.plot(time,north,'r')
ax2.xaxis.set_major_formatter(date_fmt)
ax2.xaxis.set_major_locator(months)
ax2.xaxis.set_minor_locator(days)
ax2.set_xlim([tmin,tmax])
ax3 = fig.add_subplot(gs[4:, 4:]) 
proj_max = east * u1 + north * w1
proj_min = east * u2 + north * w2
if not(name=='POO'):
    proj_max = signal.filtfilt(b, a, proj_max)
dt_stick,time_collapses,pstick,pslip = stick_slip(proj_max,time,date_caldera_start,name,date_fmt,months,days,first_day,last_day)
ax3.plot(time,proj_max,'orange')
ax3.plot(time,proj_min,'blue')
ax3.xaxis.set_major_formatter(date_fmt)
ax3.xaxis.set_major_locator(months)
ax3.xaxis.set_minor_locator(days)
ax3.set_xlim([tmin,tmax])
ax1.set_title(name)

ax = fig_main.add_subplot(2,1,1) #If you dont want period stickslip subplot
ax.axvspan(date_caldera_start,date_caldera_end,alpha = 0.3,color = 'blueviolet')
ax.axvline(x=date_eruption,)
ax.axvline(x=date_eruption_end)
ax.plot(time,proj_max, 'black')
ax.xaxis.set_major_formatter(date_fmt)
ax.xaxis.set_major_locator(fifteen_days)
ax.xaxis.set_minor_locator(days)
ax.set_xlim([first_day,last_day])
ax.set_ylabel(name + ' [microradians]',fontsize = 14,fontweight = 'heavy')

model_ERZ = LinearRegression()
model_summit = LinearRegression()


A = np.loadtxt('volumes.txt')
summit = A[0:4,:]
ERZ = A[4:,:]
t_summit = summit[:,0]
t_summit = first_day - 1 + t_summit
vol_summit = -summit[:,1]/1e+9
eb_summit = summit[:,2]
t_ERZ = ERZ[:,0]
t_ERZ = first_day - 1 + t_ERZ
vol_ERZ = ERZ[:,1]/1e+9
t_ERZ = t_ERZ.reshape(-1,1)
t_summit = t_summit.reshape(-1,1)

vol_ERZ = vol_ERZ.reshape(-1,1)
vol_summit = vol_summit.reshape(-1,1)

model_ERZ.fit(t_ERZ, vol_ERZ)
model_summit.fit(t_summit, vol_summit)

tfit = np.linspace(first_day+3,t_ERZ[-1][0])
vol_ERZ_fit = model_ERZ.predict(tfit[:, np.newaxis])
vol_summit_fit = model_summit.predict(tfit[:, np.newaxis])

ax = fig_main.add_subplot(2,1,2)
ax.axvspan(date_caldera_start,date_caldera_end,alpha = 0.3,color = 'blueviolet')

ax.plot(t_summit,vol_summit,'bo',markersize = 7 )
ax.plot(t_ERZ,vol_ERZ,'ro',markersize = 7 )
ax.legend(['Caldera collapse','Land lavas'])
ax.plot(tfit,vol_ERZ_fit,'r',linewidth = 2)
ax.plot(tfit,vol_summit_fit,'b',linewidth = 2)
ax.axvline(x=date_eruption,)
ax.axvline(x=date_eruption_end) 
ax.xaxis.set_major_formatter(date_fmt)
ax.xaxis.set_major_locator(fifteen_days)
ax.xaxis.set_minor_locator(days)
ax.set_xlim([first_day,last_day])
ax.set_ylim([0,1])
ax.set_ylabel('Volumes [km$^3$]',fontsize = 14,fontweight = 'heavy')
ax.legend(['Caldera collapses','Land lavas'])

fig_main.set_tight_layout(True)
fig_main.autofmt_xdate()
plt.figure(1)
plt.savefig('Figs/principal_component_01may_filtered.pdf')
plt.show()

