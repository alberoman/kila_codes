#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:47:41 2019

@author: aroman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:25:40 2019
Plot results from main.py 
@author: aroman
"""
import matplotlib.pyplot as plt
import pickle
import corner 
import numpy as np
import pymc3 as pm
import pandas as pd
import sys
sys.path.insert(1,'../../sar/')
from coord_conversion import *

pathgg_results = 'mu_100_lame_1e9_Rp_500/'
xSource = 84
ySource =  82
rho = 2600
g = 9.8
lame = 1e+9
poisson = 0.25
mu = 100
l = 4e+4
tiltName = 'UWD'
filename = 'DataForInversion' + tiltName + '.pickle'
data = pickle.load(open(filename,'rb'))
ref_point = np.array([-155.274,19.395])  # Ref point is alway CRIM
xy = llh2local(data['LonLat'],ref_point) *1000
radial_distance = ((xy[0] - xSource)**2 + (xy[1] - ySource)**2)**0.5
tilt_stick = data['tilt_stick'] * 1e-6
tilt_slip =  data['tilt_slip'] * 1e-6
tilt_std = data['tilt_std'] *1e-6
dt_stick = data['dt_stick'] * 3600 * 24 
x = -data['gps']

filename = 'InversionResults_mu_100UWD.pickle'
results =pickle.load(open(pathgg_results + filename,'rb'))

n = np.arange(1,len(results['tilt_slip']) + 1)
Rp = results['Rp']
S = S = 3.14 * Rp**2
R1 = rho * g * results['MAP']['V'] /(results['MAP']['k'] * S)    
x_MAP = R1 / (rho * g) * results['MAP']['pspd']/ (1 + R1) * n
phi = results['MAP']['plps'] / results['MAP']['pspd']
tau =  8 * mu *l * results['MAP']['V']/ (3.14 * results['MAP']['a']**4 * results['MAP']['k'])
dt_stick_MAP = -tau * np.log(((1 + R1) * phi - 2 * n * R1 ) / (((1 + R1) *phi - 2 *(R1*(n -1) -1)))) 
pslip_MAP = -results['MAP']['ps'] -rho * g * x_MAP
pstick_MAP= pslip_MAP + 2 * results['MAP']['pspd'] / (1 + R1)
R = (radial_distance**2 + results['MAP']['depth']**2)**0.5 # Change this if including deepth in the inversion
tilt_coeff =  9./(4*3.14) * results['MAP']['V'] * (1 - poisson)/lame* radial_distance * results['MAP']['depth']/(R**5) #Change this if inverting volume or depth
tilt_slip_MAP = pslip_MAP * tilt_coeff
tilt_stick_MAP = pstick_MAP * tilt_coeff
fig,ax = plt.subplots(nrows =3 ,ncols = 1)
ax[0].plot(n,x,'go')
ax[0].plot(n,x_MAP,'g')
ax[1].plot(n,dt_stick,'o',color = 'cyan')
ax[1].plot(n,dt_stick_MAP,color = 'cyan')
ax[2].plot(n,tilt_slip,'ro')
ax[2].plot(n,tilt_slip_MAP,'r')
ax[2].plot(n,tilt_stick,'bo')
ax[2].plot(n,tilt_stick_MAP,'b')


