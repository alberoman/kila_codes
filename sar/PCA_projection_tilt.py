#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:24:58 2019

@author: aroman
"""
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def draw_vector(v0, v1,cl, head =0,ax=None,):
    #Draws a vector with no arrow using annotation. v0 and v1 are two vectors containing the coordinates of the starting point of the arrow (v0)
    #and the final point of the arrow (v1)
    ax = ax or plt.gca()
    if head:
        arrowprops=dict(arrowstyle = '->, head_width=0.3, head_length=0.5',color= cl,
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    else:
        arrowprops=dict(arrowstyle='-',color= cl,
                    linewidth=3,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
def hillshade(array, azimuth, angle_altitude):

    # Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2

def crop_dem(array,lat_orig,lon_orig,lat_min,lat_max,lon_min,lon_max):
    array = array[(lat_orig >= lat_min) & (lat_orig <= lat_max),:]
    array = array[:,(lon_orig >= lon_min) & (lon_orig <= lon_max)]
    return array, lat_orig[(lat_orig >= lat_min) & (lat_orig <= lat_max)],lon_orig[(lon_orig >= lon_min) & (lon_orig <= lon_max)]


def PCA_vectors(x,y):
    '''
    extract principal vectors of PCA, always careful to how the vectors components are returned.
    '''
    pca = PCA(n_components=2)
    data = np.zeros((len(x),2))
    data[:,0] = x
    data[:,1] = y
    out_pca = pca.fit(data)
    umax = out_pca.components_[0,0] #First component of the vectors associated to the maximum variance
    wmax = out_pca.components_[0,1]#Second component of the vectors associated to the maximum variance
    umin = out_pca.components_[1,0] #First component of the vectors associated to the least variance
    wmin = out_pca.components_[1,1] #Second component of the vectors associated to the least variance
    return umax,wmax,umin,wmin,out_pca.mean_

        