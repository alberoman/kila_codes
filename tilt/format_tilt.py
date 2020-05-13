#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:35:48 2019

@author: aroman
Create a dictionary with the coordinates of tiltmeters, the time and the north and east component of each tiltmeter.
As each tiltmeters has different files, this script merges all the file in the same timeseries 
"""

import numpy as np 
import numpy as np
import os
import csv
import pickle 
from datetime import datetime
import matplotlib.dates as mdates



def get_data_segment(date_start,date_end,dates):
    #extract data between to dates expresses in human format
    number_start = mdates.date2num(datetime.strptime(date_start,'%m/%d/%Y'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%m/%d/%Y'))
    ind = (dates>number_start) & (dates<number_end)
    return ind,number_start,number_end

def read_csv(filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        A = []
        for row in readCSV:
            A.append(row)
    return A

def read_tilt(filename):
    A = read_csv(filename)
    date = []
    E_tilt =[]
    N_tilt = []
    count = 0
    for i in range(1,len(A)):
        date.append(A[i][0])
        E_tilt.append(float(A[i][3]))
        N_tilt.append(float(A[i][4]))
        
    date = date[1:]
    E_tilt = E_tilt[1:]
    N_tilt = N_tilt[1:]
    date_new = []
    for i in date:
        date_new.append(i[:10] + ':' +i[11:-1])
    date_obj = [datetime.strptime(date_tilt,'%Y-%m-%d:%H:%M:%S') for date_tilt in date_new]
    dates = mdates.date2num(date_obj)
    E_tilt = np.array(E_tilt)
    N_tilt = np.array(N_tilt)
    index = (np.isnan(N_tilt) == False) & (np.isnan(E_tilt) == False)
    E_tilt = E_tilt[index]
    N_tilt = N_tilt[index]
    dates = dates[index]
    return dates, E_tilt,N_tilt

def lat_lon_1station(file):
    f = open(file)
    a=f.readlines()
    str1 = a[1]
    str2 = a[2]
    latitude = float(str1.split()[1])
    longitude = float(str2.split()[1])
    f.close()
    return latitude,longitude

def lat_lon_allstations(path_data):

    file_list_txt = []
    for file in os.listdir(path_data):
        if file.endswith(".txt"):
            file_list_txt.append(file)
    coord = {}
    for i in file_list_txt:
        filename = pathgg + i
        lat,lon = lat_lon_1station(filename)
        coord[i[:3]] = np.array([lat,lon]) # each element of the dictionary is an arrays whose 
    return coord

def diff_tilt(north0,east0,north,east):
    #This is used to calculate the cumulative offset between two timeseries, calculating the difference of the second file and
    #then integrating
    diff_east = np.diff(east)
    diff_north = np.diff(north)
    cumul_north = 0
    cumul_east = 0
    east_tilt = np.zeros(len(east))
    north_tilt = np.zeros(len(east))
    for k in range(len(diff_north)):
        cumul_north = cumul_north + diff_north[k]
        cumul_east  = cumul_east + diff_east[k]
        north_tilt[k+1] =  north0 + cumul_north
        east_tilt[k+1] =  east0 + cumul_east
    east_tilt[0] = east0
    north_tilt[0] = north0
    return north_tilt,east_tilt

def extract_longest_interval(time,north,east,flag,tresh):
    # Extract the longest contigous (meaning that time interval between acquisition does not exceed a treshold 'tresh').
    dt = np.diff(time)
    while any(dt > tresh):
        index = np.where(dt > tresh)
        index =index[0][0]
        east1 = east[:index + 1]
        north1 = north[:index + 1]
        time1 = time[:index + 1]
        flag1 = flag[:index + 1]
        east2 = east[index + 1:]
        north2 = north[index + 1:]
        time2 = time[index + 1:]
        flag2 = flag[index + 1:]
        if len(time1) > len(time2):
            time = time1
            east = east1
            north = north1
            flag = flag1
        else:
            time =time2
            east = east2
            north = north2
            flag = flag2
        dt = np.diff(time)
    return time,north,east,flag

def correct_offset(north,east,flag):
    #This routine correct the offset between two files. It uses a flag vector which contains a 1 at index corresponding to the first element of each file. It is used
    #to keep track where the files have been concatenated
    while any(flag>0):
        index = np.where(flag>0)[0]
        if len(index) > 1:
            index1 = index[0]
            index2  = index[1]
            north_next,east_next = diff_tilt(north[index1 - 1],east[index1 - 1],north[index1:index2],east[index1:index2])
            north[index1:index2] = north_next
            east[index1:index2] = east_next
            flag[index1] = 0
        else:
            index1 = index[0]
            north_next,east_next = diff_tilt(north[index1 - 1],east[index1 - 1],north[index1:],east[index1:])
            north[index1:] = north_next
            east[index1:] = east_next
            flag[index1] = 0
    return north,east,flag
            
            

pathgg = '../../data/Tiltmeterdatafr/'
day_beginning =  '07/01/2018'
day_end =  '08/15/2018'

station_coord = lat_lon_allstations(pathgg)
station = {} # Initialize a dictionary for the station
for station_name in station_coord.keys():
    station[station_name] = {}
    station[station_name]['lon_lat'] = np.array([station_coord[station_name][1],station_coord[station_name][0]])

file_list = []
for file in os.listdir(pathgg):
    if file.endswith(".csv"):
       file_list.append(file)
file_list = sorted(file_list)
file_list_truncated = []
for i in file_list:
    file_list_truncated.append(i[0:3])
stations = list(set(file_list_truncated))
N_stations = len(stations)
station_indexes = {}
for i in stations:
    station_indexes[i] = [j for j,value in enumerate(file_list_truncated) if value == i]
threshold_time = 2
#station_indexes_temp = {'SDH': station_indexes['SDH']}
#station_indexes = station_indexes_temp
for i in station_indexes.keys():
    print('Formatting station ',i)
    t_full = np.array([])
    east_full = np.array([])
    north_full = np.array([])
    flag_full = np.array([])            #flag array to keep track of the concatenation of different files
    for j in range(len(station_indexes[i])):
        time,east_tilt,north_tilt = read_tilt(pathgg + file_list[station_indexes[i][j]])
        
        plt.figure()
        plt.plot(time,east_tilt,'r',time,north_tilt,'b')
        plt.ylabel(i)
        flag = np.zeros(len(time))
        flag[0] = 1    #the one in each array correspond to the first element of each file
        east_full = np.concatenate((east_full,east_tilt))
        north_full = np.concatenate((north_full,north_tilt))
        t_full = np.concatenate((t_full,time))
        flag_full = np.concatenate((flag_full,flag))
    flag_full[0] = 0       #the last element of the flag is not a concatenation!
    index,start,end = get_data_segment(day_beginning,day_end,t_full)
    east_full = east_full[index]
    north_full = north_full[index]
    t_full = t_full[index]
    flag_full = flag_full[index]
    t_short,north_short,east_short,flag_short = extract_longest_interval(t_full,north_full,east_full,flag_full,threshold_time)
    north_fixed,east_fixed,flag_fixed=  correct_offset(north_short,east_short,flag_short)
    station[i]['time'] = t_short
    station[i]['north'] = north_fixed
    station[i]['east'] = east_fixed
pickle.dump(station, open( "tilt_dictionary_01july.pickle", "wb" ))

    
        