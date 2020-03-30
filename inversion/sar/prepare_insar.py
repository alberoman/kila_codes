#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:13:07 2019

@author: aroman
"""
import sys
import pickle
sys.path.insert(1,'../../sar/')
import numpy as np
from coord_conversion import *

def geometry_vector(lat_geom,lon_geom,lat,lon,var):
    var_vector = np.zeros(lat.shape)
    for counter,dummy in enumerate(lat):
        ind = np.argmin(np.sqrt((lat[counter] - lat_geom)**2 + (lon[counter] - lon_geom)**2))
        var_vector[counter] = var[0,ind]
    return var_vector 

def prepare_data(path_data,filename,ref_point,):
    nPoints = 0
    LonLat = np.zeros((0,2))
    interfero = pickle.load(open(path_data + filename,'rb'))
    wavelength = np.array(interfero['wavelength'])
    
    phase = np.array(interfero['phase'])
    lat  = np.float32(np.array(interfero['lat']))
    lon = np.float32(np.array(interfero['lon']))
    #lat = lat[phase < 0] # Careful here we are considering only elements showing deflation
    #lon = lon[phase < 0]
    #phase = phase[phase < 0]
    lat= lat[np.logical_not(np.isnan(phase))]
    lon= lon[np.logical_not(np.isnan(phase))]
    phase = phase[np.logical_not(np.isnan(phase))]
    los =  phase / (4 * 3.14) *  wavelength
    LonLat= np.concatenate((lon.reshape(1,lon.size),lat.reshape(1,lon.size)),0)
    
    xy = llh2local(LonLat, ref_point)
    nPointsVector = np.arange(1,len(lon) + 1)
    nPointsVector.shape = (len(nPointsVector),1)
    xy = np.concatenate((nPointsVector,xy.transpose() * 1000),1)
    
    '''
    Geometry parameters. This is valid as long as not using downsamplig (eg quadtree).
    If adding downsampling this part of the code has to be modified.
    '''
    
    inc = interfero['inc']
    look = interfero['look']
    azi = interfero['azi']
    latMeta = interfero['latMeta']
    lonMeta = interfero['lonMeta']
    LONM,LATM =  np.meshgrid(lonMeta,latMeta)
    inc = inc.reshape(1,inc.size)
    look = look.reshape(1,look.size)
    azi = azi.reshape(1,azi.size)
    lonMeta = LONM.reshape(1,LONM.size)
    latMeta = LATM.reshape(1,LATM.size)
    var_vector = np.zeros(lat.shape)
    inc  = geometry_vector(latMeta,lonMeta,lat,lon,inc)
    look  = geometry_vector(latMeta,lonMeta,lat,lon,look)
    azi  = geometry_vector(latMeta,lonMeta,lat,lon,azi)
    pts = {}
    nb = len(lat)
    pts['xy'] =  xy
    pts['LonLat'] = np.transpose(LonLat)
    insar = {}
    insar['obs'] = pts['xy'][:,1:]
    insar['los'] = los
    insar['look'] = look
    insar['azi'] = azi
    insar['inc'] = inc
    insar['LonLat'] = LonLat
    insar['ref_coord'] = ref_point
    obs = pts['xy'][:,1:]
    X1,X2 = np.meshgrid(obs[:,0],obs[:,0])
    Y1,Y2 = np.meshgrid(obs[:,1],obs[:,1])
    H = np.sqrt((X1-X2)**2 + (Y1 - Y2)**2) # Calculate distance between points
    '''
    This is only valid if not calculating variograms
    '''
    insar['sillExp'] = 0.04 ** 2
    insar['nugget'] = 0.002 ** 2
    insar['range'] = 5000
    covarianceMatrix = insar['sillExp'] * np.exp(-H/insar['range']) + insar['nugget'] * np.identity(nb) # Calculate covariance matrix for exponential model with nugget
    insar['cov'] = covarianceMatrix 
    insar['InvCov'] = np.linalg.inv(covarianceMatrix)
    obs = np.concatenate((obs.transpose(),np.zeros((1,phase.size))),0)
    nObs = phase.size
    
    '''
    Preparation finished
    '''
    return insar





