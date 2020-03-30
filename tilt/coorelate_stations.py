#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the linear correlation between statiosn

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
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import signal
from PCA_projection_tilt import *
from scipy.interpolate import *
from sklearn.linear_model import LinearRegression


plt.rcParams['svg.fonttype'] = 'none'

stations = pickle.load(open('tilt_dictionary_20june.pickle','rb')) 
'''
Find mininimum and maximum time for colormap
'''
tmin = 0
tmax = 1e+16
b, a = signal.butter(2, 0.03)
months = mdates.MonthLocator()  # every month
fifteen_days = mdates.DayLocator(bymonthday=(1,15))  # labels every 15 day
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')
list_station = ['UWD','SDH','POO','JKA']  #Select station to plot


day_beginning = '2018-05-01' #all timeseries
day_end = '2018-08-06'
first_day = mdates.date2num(datetime.strptime(day_beginning,'%Y-%m-%d'))
last_day = mdates.date2num(datetime.strptime(day_end,'%Y-%m-%d'))

for name in list_station:
    east = stations[name]['east']
    north = stations[name]['north']
    east = east - east[0]
    nort = north - north[0]
    tilt = (north**2 + east**2)**0.5 
    stations[name]['tilt'] = tilt
    time = stations[name]['time']
    t0  = stations[name]['time'][0]
    tend = stations[name]['time'][-1]
    tmin = max(tmin,t0)
    tmax = min(tmax,tend)
N = (tmax - tmin) / 5e-4
time_ref =  np.linspace(tmin,tmax,N)
for name in list_station:
    f = interp1d(stations[name]['time'],stations[name]['tilt'])
    stations[name]['tilt'] = f(time_ref)
for name in list_station:
    reg = LinearRegression()
    reg.fit(stations['SDH']['tilt'],stations[name]['tilt'])
    correlation = reg.score(stations['SDH']['tilt'],stations[name]['tilt'])
    text = 'correlation coeff between SDH and ' + name + 'is ' + str(correlation)
    print(text)
    
    
    


