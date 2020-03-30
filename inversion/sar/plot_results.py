#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:25:40 2019
Plot results from main.py 
@author: aroman
"""
import matplotlib.pyplot as plt
import sys
plt.rcParams['svg.fonttype'] = 'none'
sys.path.insert(1,'../../sar/')
from coord_conversion import *
import pickle
import corner 
import numpy as np
import pymc3 as pm
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LightSource
import gdal
from prepare_insar_paul import prepare_data


def crop_dem(array,lat_orig,lon_orig,lat_min,lat_max,lon_min,lon_max):
    array = array[(lat_orig >= lat_min) & (lat_orig <= lat_max),:]
    array = array[:,(lon_orig >= lon_min) & (lon_orig <= lon_max)]
    return array, lat_orig[(lat_orig >= lat_min) & (lat_orig <= lat_max)],lon_orig[(lon_orig >= lon_min) & (lon_orig <= lon_max)]
def read_paul_data(path_data,fname):
    filename1 = path_data + 'QuadCov'+ fname
    filename2 = path_data + 'Quad' + fname
    covariance = np.loadtxt(path_data + filename1)
    data = np.loadtxt(path_data + filename2)
    vel = data[:,0]
    lon = data[:,2]
    lat = data[:,3]
    incidence = data[:,-2]
    look_angle = data[:,-1]
    return lon,lat,vel,incidence,look_angle

def direct_mogi_model(xSource,ySource,depthSource,strSource,data):
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
    R = np.sqrt((x - xSource)**2 + (y - ySource)**2 + depthSource **2)
    coeff_east = (x - xSource)/ R**3 * elastic_constant
    coeff_north = (y - ySource)/ R**3 * elastic_constant
    coeff_up = depthSource / R**3 * elastic_constant
    east = strSource * coeff_east
    north = strSource * coeff_north
    vert= strSource * coeff_up
    Ulos = east * UEast + north * UNorth + vert * UVert
    return Ulos
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
    return Ulos
    
def plot_scatter_ondem(lons_data,lats_data,val,pathgg,vname,lonSource1 = 0,latSource1 =0,lonSource2 = 0,latSource2 =0):
    verticalEx = 0.1    
    ls = LightSource(azdeg=135, altdeg=30)
    
    lat_minimum = 19.31
    lat_maximum = 19.48
    lon_minimum = -155.40
    lon_maximum = -155.15
    
    lat_minimum_DEM = 19.31
    lat_maximum_DEM = 19.48
    lon_minimum_DEM = -155.40
    lon_maximum_DEM = -155.15
    ds = gdal.Open(path_data + 'dem_srtm.tif')
    data = ds.ReadAsArray()
    gt=ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    minx = gt[0]
    miny = gt[3] + cols*gt[4] + rows*gt[5] 
    maxx = gt[0] + cols*gt[1] + rows*gt[2]
    maxy = gt[3]
    lons = np.linspace(minx,maxx,cols)
    lats = np.linspace(miny,maxy,rows)
    lats = lats[::-1]
    #elats = lats[::-1]
    dem,lat_DEM,lon_DEM = crop_dem(data,lats,lons,lat_minimum_DEM,lat_maximum_DEM,lon_minimum_DEM,lon_maximum_DEM)    #Cropping the DEM
    rgb = ls.shade(dem, cmap=plt.cm.gist_gray, vert_exag=verticalEx, blend_mode='overlay')
    '''
    Defining parameters for DEM plot
    NOTE: as the DEM is plotted with imshow (which is in pixel coordinates), the llclon... parameters 
    must correspond to the corner of the matrixm otherwise georeferencing and subsequent coordinate plot DO NOT WORK!!! 
    '''
    llclon = np.min(lon_DEM)
    llclat = np.min(lat_DEM)
    urclon = np.max(lon_DEM)
    urclat = np.max(lat_DEM)
    
    map = Basemap(projection='tmerc', 
                  lat_0=(llclat+urclat) * 0.5, lon_0=(llclon+urclon) * 0.5,
                  llcrnrlon=llclon, 
                  llcrnrlat=llclat, 
                  urcrnrlon=urclon, 
                  urcrnrlat=urclat)
    lons_map,lats_map = map(lons_data,lats_data)
   

    im1 = map.imshow(rgb,origin = 'upper',rasterized =True)
    im2 = map.scatter(lons_map,lats_map,s = 20,c = val,alpha = 0.4 ,cmap = 'jet', vmin = -4, vmax = 0 )
    if lonSource1:
        lonSource_map1,latSource_map1 = map(lonSource1,latSource1)
        im3 = map.scatter(lonSource_map1,latSource_map1,s = 20,c = 'black')
    if lonSource2:
        lonSource_map2,latSource_map2 = map(lonSource2,latSource2)
        im3 = map.scatter(lonSource_map2,latSource_map2,s = 20,c = 'black')
    cbar = map.colorbar(im2,shrink = 0.2,alpha = 1)
    cbar.set_label('LOS velocity [m/year]',fontsize = 18)
    cbar.solids.set_rasterized(True)
    plt.savefig(pathgg + vname + '.pdf',dpi = 300)
    plt.close('all')
    
ref_coord = np.array([-155.274,19.395]) #Cordinate of CRIM station
  
fname_asc = '_t124.out'
fname_des = '_t87.out'

filename = 'Mogi_Metropolis_100000_2mogi.pickle'
path_data_sar  = '../../data/sar/'
pathgg_results = 'results/2mogi/'
path_data  = '../../data/'
path_Figs  = 'Figs/2mogi/'




results =pickle.load(open(pathgg_results + filename,'rb'))
coord_source1 = np.zeros((2,1))
coord_source1[0,0] = results['MAP']['xSource1'] / 1000
coord_source1[1,0] = results['MAP']['ySource1'] / 1000

coord_source2 = np.zeros((2,1))
coord_source2[0,0] = results['MAP']['xSource2'] / 1000
coord_source2[1,0] = results['MAP']['ySource2'] / 1000
lonlatSource1 = local2llh(coord_source1,ref_coord)
lonlatSource2 = local2llh(coord_source2,ref_coord)

panda_trace = pm.backends.tracetab.trace_to_dataframe(results['trace'])
fig1 = plt.figure(1)
corner.corner(panda_trace[['depthSource1','xSource1','ySource1','depthSource2','xSource2','ySource2']],truths = [results['MAP']['depthSource1'],results['MAP']['xSource1'],results['MAP']['ySource1'],
                                                                                                                 results['MAP']['depthSource2'],results['MAP']['xSource2'],results['MAP']['ySource2']])
plt.savefig(path_Figs + 'hist_SAR.pdf')
plt.close('all')
#First track
lon_data,lat_data, v, inc, look = read_paul_data(path_data_sar,fname_asc)
plot_scatter_ondem(lon_data,lat_data,v,path_Figs,'data_' + fname_asc)
d = prepare_data(ref_coord,path_data_sar,fname_asc)
#vModel = direct_mogi_model(results['MAP']['xSource'],results['MAP']['ySource'],results['MAP']['depthSource'],results['MAP']['strSource'],d)
vModel = direct_2mogi_model(results['MAP']['xSource1'],results['MAP']['ySource1'],results['MAP']['depthSource1'],results['MAP']['strSource1'],
                           results['MAP']['xSource2'],results['MAP']['ySource2'],results['MAP']['depthSource2'],results['MAP']['strSource2'],d)
plot_scatter_ondem(lon_data,lat_data,vModel,path_Figs,'model_' + fname_asc,lonlatSource1[0],lonlatSource1[1],lonlatSource2[0],lonlatSource2[1])
residual = v - vModel
plot_scatter_ondem(lon_data,lat_data,residual,path_Figs,'residual_' + fname_asc,lonlatSource1[0],lonlatSource1[1],lonlatSource2[0],lonlatSource2[1])
#Second track
lon_data,lat_data, v, inc, look = read_paul_data(path_data_sar,fname_des)
plot_scatter_ondem(lon_data,lat_data,v,path_Figs,'data_' + fname_des)
d = prepare_data(ref_coord,path_data_sar,fname_des)
#vModel = direct_mogi_model(results['MAP']['xSource'],results['MAP']['ySource'],results['MAP']['depthSource'],results['MAP']['strSource'],d)
vModel = direct_2mogi_model(results['MAP']['xSource1'],results['MAP']['ySource1'],results['MAP']['depthSource1'],results['MAP']['strSource1'],
                           results['MAP']['xSource2'],results['MAP']['ySource2'],results['MAP']['depthSource2'],results['MAP']['strSource2'],d)
plot_scatter_ondem(lon_data,lat_data,vModel,path_Figs,'model_' + fname_des,lonlatSource1[0],lonlatSource1[1],lonlatSource2[0],lonlatSource2[1])
residual = v - vModel
plot_scatter_ondem(lon_data,lat_data,residual,path_Figs,'residual_' + fname_des,lonlatSource1[0],lonlatSource1[1],lonlatSource2[0],lonlatSource2[1])








    








 

