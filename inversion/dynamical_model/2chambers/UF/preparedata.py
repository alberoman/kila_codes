#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare data tilt and GPS for inversion

@author: aroman
"""

import sys
sys.path.insert(1,'../../../../sar/')
import pickle
import numpy as np 
import numpy as np
from coord_conversion import *
import csv
import matplotlib.dates as mdates
from datetime import datetime
import pymc3 as pm

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

def preparation(list_station,date,factor,mt):
    path_data = '../../../../../data/'

    stations = pickle.load(open(path_data + 'tilt_dictionary_01may.pickle','rb'))
    if date == '06-16-2018':
        t0 = 736861.85 # This is night of 06/16/2018
    else:
        t0 = 736878.53
    
    tend = 736910
    ref_point = np.array([-155.274,19.395]) #Cordinate of CRIM station
    dg2rd = np.pi / 180
    tx = []
    ty = []
    t = []
    x = []
    y = []
    nstation = []
    sample = 24
    i = 0
    for name in list_station:
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
        tx.append(east*1e-6)
        ty.append(north*1e-6)
        t.append(time)
        lonTemp = stations[name]['lon_lat'][0] * np.ones(len(east))   
        latTemp = stations[name]['lon_lat'][1] * np.ones(len(east))
        LonLat= np.concatenate((lonTemp.reshape(1,lonTemp.size),latTemp.reshape(1,latTemp.size)),0)
        xy = llh2local(LonLat, ref_point)
        xy = xy.transpose() * 1000
        x.append(xy[:,0])
        y.append(xy[:,1])
        nstation.append(i * np.ones(len(east)))
        #axx = fig.add_subplot(211)
        #axx.plot(time,north)
    
        #axy = fig.add_subplot(212)
        #axy.plot(time,east)
        i = i + 1
    #axx.set_ylabel('north')
    #axy.set_ylabel('east')
    
    x = np.array(x)
    x = np.concatenate(x)
    
    y = np.array(y)
    y = np.concatenate(y)
    
    tx = np.array(tx)
    tx = np.concatenate(tx)
    
    ty = np.array(ty)
    ty = np.concatenate(ty)
    
    t = np.array(t) * 3600*24
    tTilt = np.concatenate(t)
    nstation = np.array(nstation)
    nstation = np.concatenate(nstation)
    nstation = nstation.astype(int)
    '''
    Prep GPS 
    '''
    gps = 'CALS.csv'
    data = read_csv(path_data+ 'GPS/' + gps)
    time,date,disp = format_gps(data)
    date_new = []
    for line in date:
        date_temp = line[:10]
        date_new.append(mdates.date2num(datetime.strptime(date_temp,'%Y-%m-%d')))
    tGPS = np.array(date_new)
    GPS = disp
    
    GPS = GPS[tGPS>t0]
    tGPS = tGPS[tGPS > t0]
    GPS = GPS[tGPS < tend]
    tGPS = tGPS[tGPS < tend]
    tGPS = tGPS - t0
    tGPS = tGPS * 3600 * 24
    '''
    Source locations from sar
    '''
    filename = 'Mogi_Metropolis_100000_2mogi.pickle'
    results =pickle.load(open(path_data + filename,'rb'))
    panda_trace = pm.backends.tracetab.trace_to_dataframe(results['trace'])
    if mt == 'UF':
        dde = panda_trace.mean()['depthSource2']
        xde = panda_trace.mean()['xSource2']
        yde = panda_trace.mean()['ySource2']
        
        dsh = panda_trace.mean()['depthSource1']
        xsh = panda_trace.mean()['xSource1']
        ysh = panda_trace.mean()['ySource1']
        
        ddeErr = panda_trace.std()['depthSource2']
        xdeErr = panda_trace.std()['xSource2']
        ydeErr = panda_trace.std()['ySource2']
        
        dshErr = panda_trace.std()['depthSource1']
        xshErr = panda_trace.std()['xSource1']
        yshErr = panda_trace.std()['ySource1']
    elif mt == 'LF':
        dde = panda_trace.mean()['depthSource1']
        xde = panda_trace.mean()['xSource1']
        yde = panda_trace.mean()['ySource1']
        
        dsh = panda_trace.mean()['depthSource2']
        xsh = panda_trace.mean()['xSource2']
        ysh = panda_trace.mean()['ySource2']
        
        ddeErr = panda_trace.std()['depthSource1']
        xdeErr = panda_trace.std()['xSource1']
        ydeErr = panda_trace.std()['ySource1']
        
        dshErr = panda_trace.std()['depthSource2']
        xshErr = panda_trace.std()['xSource2']
        yshErr = panda_trace.std()['ySource2']
    
    locTruth =np.array([xsh,ysh,dsh,xde,yde,dde])
    locErr =np.array([xshErr,yshErr,dshErr,xdeErr,ydeErr,ddeErr])
    

    locErr = locErr * factor
    
    tx = - tx
    ty =  -ty
    GPS = -GPS
    Nst = 1 + np.max(nstation)
    return Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0
