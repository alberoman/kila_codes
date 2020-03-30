#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:42:39 2020

@author: aroman
"""
import sys
import pickle
sys.path.insert(1,'../../sar/')
import numpy as np
from coord_conversion import *
def prepare_data(ref_point,path_data,fname):
    filename1 = 'QuadCov'+ fname
    filename2 = 'Quad' + fname
    covariance = np.loadtxt(path_data + filename1)
    data = np.loadtxt(path_data + filename2)
    vel = data[:,0]
    lon = data[:,2]
    lat = data[:,3]
    inc = data[:,-2]
    look = data[:,-1]
    heading = - 360 - look 
    LonLat= np.concatenate((lon.reshape(1,lon.size),lat.reshape(1,lon.size)),0)
    xy = llh2local(LonLat, ref_point)
    nPointsVector = np.arange(1,len(lon) + 1)
    nPointsVector.shape = (len(nPointsVector),1)
    xy = np.concatenate((nPointsVector,xy.transpose() * 1000),1)
    pts = {}
    insar = {}
    nb = len(lat)
    pts['xy'] =  xy
    pts['LonLat'] = np.transpose(LonLat)
    insar = {}
    insar['obs'] = pts['xy'][:,1:]
    insar['vel'] = vel
    insar['heading'] = heading
    insar['inc'] = inc
    insar['LonLat'] = LonLat
    insar['ref_coord'] = ref_point
    insar['cov'] = covariance 
    insar['InvCov'] = np.linalg.inv(covariance)
    return insar

