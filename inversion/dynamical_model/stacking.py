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
from scipy.optimize import curve_fit
import pymc3 as pm

def double_exp(t,A,B,E,T1,phi,condd):
    Vs = 10**9.5
    ks = 10**8
    ld = 2500
    mu = 100
    tstar = 8 * Vs * mu * ld / (3.14 * ks *100)
    
    y  = A * np.exp(t/tstar*(-T1/2 - phi/2 + np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + B * np.exp(t/tstar*(-T1/2 - phi/2 - np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - E
    return y

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


def stick_slip(tilt_filt,dates):
  
    dist = 500
    prom = 10
        
    ind, _ = signal.find_peaks(tilt_filt,distance = dist,prominence= prom)
    time_peaks = dates[ind]
    peaks = tilt_filt[ind]
  
    return time_peaks,peaks,ind
date = '07-03-2018'

if date == '06-16-2018':
    t0 = 736861.85 # This is night of 06/16/2018
else:
    t0 = 736878.53
name = 'UWD' #Tilt station to invert
sample = 1
tend = 736910

path_data = '../../../data/'
path_GPS = path_data + 'GPS/'
stations = pickle.load(open(path_data + 'tilt_dictionary_01may.pickle','rb')) 
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
time = time[::sample]
east = east[::sample]
north = north[::sample]
north = north[time > t0]
east = east[time > t0]
time = time[time > t0]
north = north[time < tend]
east = east[time < tend]
time = time[time < tend]
time = time - t0
east = east - east[0]
north = north - north[0]

timebase = np.linspace(0,0.2,200)
timebase = np.concatenate((timebase,np.linspace(0.1,1,100)))
north_filt = signal.filtfilt(b, a,north)
time_peaks,peaks,ind = stick_slip(north_filt,time)
stack  =[]
dtiltslipnorth = []
dtiltslipeast = []
dtiltsticknorth = []
dtiltstickeast = []

counter = 0
for i in range(len(ind)-1):
    north_cyc =  north[ind[i]:ind[i +1]-50] -  north[ind[i]]
    east_cyc =  north[ind[i]:ind[i +1]-50] -  east[ind[i]]
    time_cyc = time[ind[i]:ind[i + 1]-50] - time[ind[i]]
    dtiltslipnorth.append(north[ind[i]-50])
    dtiltslipeast.append(east[ind[i]-50])
    dtiltsticknorth.append(north[ind[i]])
    dtiltstickeast.append(east[ind[i]])
    if time_cyc[-1] > 1:
        tiltint = np.interp(timebase,time_cyc,east_cyc)

        stack.append(tiltint)
        counter = counter + 1
dtiltstnorth = np.array(dtiltsticknorth)
tiltstnorth = -dtiltstnorth*1e-6
dtiltsteast = np.array(dtiltstickeast)
tiltsteast = -dtiltsteast * 1e-6
dtiltslnorth = np.array(dtiltslipnorth)
tiltslnorth = -dtiltslnorth*1e-6
dtiltsleast = np.array(dtiltslipeast)
tiltsleast = -dtiltsleast*1e-6
tiltstnorth = tiltstnorth - tiltslnorth[0]
tiltsteast = tiltsteast - tiltsleast[0]
tiltslnorth = tiltslnorth - tiltslnorth[0]
tiltsleast = tiltsleast - tiltsleast[0]

dtslip = np.diff(time_peaks) * 2600 * 24
stack = np.array(stack)
stack = np.mean(stack,axis = 0)
stackspr = np.std(stack,axis = 0)
timebase = timebase * 3600 * 24
popt, pcov = curve_fit(double_exp, timebase, stack)
plt.figure()
plt.fill_between(timebase,stack-stackspr,stack+stackspr)
plt.plot(timebase,stack,'co')
plt.plot(timebase,double_exp(timebase,*popt),'y')
plt.figure()
#plt.plot(time,north)
plt.plot(time_peaks[:-1],tiltslnorth,'ro')
plt.plot(time_peaks[:-1],tiltstnorth,'bo')
plt.figure()
#plt.plot(time,east)
plt.plot(time_peaks[:-1],tiltsleast,'ro')
plt.plot(time_peaks[:-1],tiltsteast,'bo')
gps_list = ['CALS.csv']
for gps in gps_list:
    data = read_csv(path_GPS + gps)
    time_gps,date,disp = format_gps(data)
    date_new = []
    for line in date:
        date_temp = line[:10]
        date_new.append(mdates.date2num(datetime.strptime(date_temp,'%Y-%m-%d')))
    date_gps = np.array(date_new)
date_gps = date_gps - t0
gpsinterp = np.interp(time_peaks,date_gps,disp)
gps = -gpsinterp
gps = gps -gps[0]
gps= gps[:-1]
plt.figure()
plt.plot(time_peaks[:-1],gps)
tstack = timebase
'''
Source locations from sar
'''
filename = 'Mogi_Metropolis_100000_2mogi.pickle'
results =pickle.load(open(path_data + filename,'rb'))
panda_trace = pm.backends.tracetab.trace_to_dataframe(results['trace'])


dsh = panda_trace.mean()['depthSource1']
xsh = panda_trace.mean()['xSource1']
ysh = panda_trace.mean()['ySource1']



dshErr = panda_trace.std()['depthSource1']
xshErr = panda_trace.std()['xSource1']
yshErr = panda_trace.std()['ySource1']

pickle.dump([dtslip,tiltstnorth,tiltsteast,tiltslnorth,tiltsleast,gps,stack,tstack,xsh,ysh,dsh,xshErr,yshErr,dshErr],open('data2ch.pickle','wb'))
    

