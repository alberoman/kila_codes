#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:50:26 2019

@author: alb
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
  
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        A = []
        for row in readCSV:
            A.append(row)
    return A

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

pathgg = '/Users/aroman/work/kilauea_2018/'
gps_list = ['gps3.csv','gps1.csv','gps2.csv']
A = np.loadtxt(pathgg + 'tilt.txt')
time_tilt = A[:,0]
tilt = A[:,6]
tmin = np.min(time_tilt)
tmax = 0
for gps in gps_list:
    data = read_csv(pathgg + gps)
    time,date,disp = format_gps(data)
    tmin = np.min([np.min(time),tmin])
counter = 0
fig1, ax1 = plt.subplots(nrows = len(gps_list) + 1, ncols = 1, figsize = (16,12))
fig2, ax2 = plt.subplots(nrows = len(gps_list), ncols = 1, figsize = (16,12)) 
 
for gps in gps_list:
    data = read_csv(pathgg + gps)
    time,date,disp = format_gps(data)
    time = time - tmin
    time = time /1e+6
    tmax = np.max([np.max(time),tmax])
    ax1[counter].plot(time,disp - disp[0])
    ax1[counter].set_ylabel(data[4][2] + ' m')
    ax2[counter].plot(time[:-1],np.diff(disp)/np.diff(time)*3600*24)
    ax2[counter].set_ylabel(data[4][2] + ' m/day')
    counter = counter + 1

time_tilt = time_tilt - tmin
time_tilt = time_tilt /1e+6
tmax = np.max([np.max(time_tilt),tmax])

ax1[counter].plot(time_tilt,tilt)
ax1[counter].set_ylabel('tilt')
for i in range(len(ax1)):
    ax1[i].set_xlim([0,10])
for i in range(len(ax2)):
    ax2[i].set_xlim([0,10])

