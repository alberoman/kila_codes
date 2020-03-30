#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:52:35 2019

@author: aroman
"""

import numpy as np
import pickle
import os,glob
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
def GPS_one_station(pathfile,date_start,date_end):
    east = []
    north = []
    up = []
    t = []
    f = open(pathfile,'r')
    lines = f.readlines()
    lines = lines[1:]
    data = {}
    for string in lines:
        string_splitted = string.split()
        t_temp = string_splitted[1]
        t_temp = t_temp[:2] + '-' + t_temp[2:5] + '-' +t_temp[5:]
        t.append(t_temp)
        east.append(float(string_splitted[8]))
        north.append(float(string_splitted[10]))
        up.append(float(string_splitted[12]))
    
    date_obj = [datetime.strptime(date_GPS,'%y-%b-%d') for date_GPS in t]
    t_number = mdates.date2num(date_obj)
    number_start = mdates.date2num(datetime.strptime(date_start,'%Y-%m-%d'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%Y-%m-%d'))
    ind = (t_number > number_start) & (t_number < number_end)
    t_number = t_number[ind]
    data['date'] = t_number
    data['east'] = np.array(east)[ind]
    data['north'] = np.array(north)[ind]
    data['up'] = np.array(up)[ind]
    data['lon_meters'] = float(string_splitted[7])
    data['lat_meters'] = float(string_splitted[9])
    data['elevation'] = float(string_splitted[11])
    
    return data 
        

pathgg = 'nevada/'
filelist = glob.glob(pathgg + '*.txt')
component_plot = 'east'
GPS = {}

first_day = '2018-07-04'
last_day = '2018-07-28'
months = mdates.MonthLocator()  # every month
fifteen_days = mdates.DayLocator(bymonthday=(1,15))  # labels every 15 day
days = mdates.DayLocator()  # every day
date_fmt = mdates.DateFormatter('%m/%d')

for filename in filelist:
    data = GPS_one_station(filename,first_day,last_day)
    GPS[filename[7:-4]] = data
fig = plt.figure()
ax = fig.add_subplot()   
for station in GPS.keys():
    ax.plot(GPS[station]['date'],GPS[station][component_plot]-GPS[station][component_plot][0],'o')
    ax.text(GPS[station]['date'][-1],GPS[station][component_plot][-1]-GPS[station][component_plot][0],station)
ax.xaxis.set_major_formatter(date_fmt)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
plt.show()
pickle.dump(GPS, open( pathgg + "dict_GPS.pickle", "wb" ))
        
        
        