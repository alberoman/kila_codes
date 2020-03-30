#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:05:21 2020

@author: aroman
"""


import numpy as np
from prepare_insar_paul import prepare_data

def direct_2mogi_model(xSource1,ySource1,depthSource1,strSource1,xSource2,ySource2,depthSource2,strSource2,data):
    dg2rd = np.pi / 180
    poisson = 0.25
    lame = 1e+9
    elastic_constant = (1 - poisson) / lame
    x = data['obs'][:,0]
    y = data['obs'][:,1]
    lon = data['LonLat'][:,0]
    lat = data['LonLat'][:,1]
    inc = data['inc'] * dg2rd
    heading = data['heading'] * dg2rd
    UEast = -np.cos(heading) * np.sin(inc) 
    UNorth = np.sin(heading) * np.sin(inc) 
    UVert = np.cos(inc)
    R1 = np.sqrt((x - xSource1)**2 + (y - ySource1)**2 + depthSource1 **2)
    R2 = np.sqrt((x - xSource2)**2 + (y - ySource2)**2 + depthSource2 **2)
    coeff_east1 = (x - xSource1)/ R1**3 * elastic_constant
    coeff_east2 = (x - xSource2)/ R2**3 * elastic_constant
    coeff_north1 = (y - ySource1)/ R1**3 * elastic_constant
    coeff_north2 = (y - ySource2)/ R2**3 * elastic_constant
    coeff_up1 = depthSource1 / R1**3 * elastic_constant
    coeff_up2 = depthSource2 / R2**3 * elastic_constant
    
    east1 = strSource1 * coeff_east1
    east2 = strSource2 * coeff_east2
    north1 = strSource1 * coeff_north1
    north2 = strSource2 * coeff_north2
    vert1= strSource1 * coeff_up1
    vert2= strSource2 * coeff_up2
    east = east1 + east2
    north = north1 + north2
    vert = vert1 + vert2
    Ulos = east * UEast + north * UNorth + vert * UVert
    return x,y,vert

def add_noise_gps()


xSource1 = 0
ySource1 = 0
xSource2 = 2000
ySource2 = -2000
depthSource1 = 1000
depthSource2 = 3000
V1 = 1e+9
V2 = 1e+10
DeltaP1 = 2e+6
DeltaP2 = 1e+6
strSource1 = V1 * DeltaP1
strSource2 = V2 * DeltaP2

filename = '_t124.out'
data = prepare_data(ref_coord,pathgg_data,filename)
x_asc = data['obs'][:,0]
y_asc = data['obs'][:,1]
lon_asc = data['LonLat'][:,0]
lat_asc = data['LonLat'][:,1]
covariance_asc = data['cov']
inc_asc = data['inc'] * dg2rd
heading_asc = data['heading'] * dg2rd
UEast_asc = -np.cos(heading_asc) * np.sin(inc_asc) 
UNorth_asc = np.sin(heading_asc) * np.sin(inc_asc) 
UVert_asc = np.cos(inc_asc)
U_asc = data['vel']
#Setting some bounds
xmin_asc = np.min(x_asc)
xmax_asc = np.max(x_asc)
ymin_asc = np.min(y_asc)
ymax_asc = np.max(y_asc)
x_asc,y_asc,uz_asc = direct_2mogi_model(xSource1,ySource1,depthSource1,strSource1,xSource2,ySource2,depthSource2,strSource2,data)

filename = '_t87.out'
data = prepare_data(ref_coord,pathgg_data,filename)
x_des = data['obs'][:,0]
y_des = data['obs'][:,1]
lon_des = data['LonLat'][:,0]
lat_des = data['LonLat'][:,1]
covariance_des = data['cov']
inc_des = data['inc'] * dg2rd
heading_des = data['heading'] * dg2rd
UEast_des = -np.cos(heading_des) * np.sin(inc_des) 
UNorth_des = np.sin(heading_des) * np.sin(inc_des) 
UVert_des = np.cos(inc_des)
U_des = data['vel']
#Setting some bounds
xmin_des = np.min(x_des)
xmax_des = np.max(x_des)
ymin_des = np.min(y_des)
ymax_des = np.max(y_des)

xSource1_model = 0
ySource1_model = 0
depthSource1_model = 0    