#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:42:00 2019

@author: aroman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:37:05 2019

@author: aroman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:17:50 2019

@author: aroman
"""
import numpy as np
import theano
import theano.tensor as tt
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import pymc3 as pm
import pickle as pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'../../sar/')
from coord_conversion import *


g = 9.8
rho = 2700
#chamber and elasticity
#Elastic properties crust
poisson = 0.25
lame = 1e+9
#Position of the chamber
#Piston geometrical parameters
Rp = 7e+2
S = 3.14 * Rp**2
#Conduit Properties
l = 4e+4
mu =  100
tiltName = 'SDH'
filename = 'DataForInversion' + tiltName + '.pickle'
data = pickle.load(open(filename,'rb'))
tilt_stick = data['tilt_stick'] * 1e-6
tilt_slip =  data['tilt_slip'] * 1e-6
tilt_std = data['tilt_std'] *1e-6
dt_stick = data['dt_stick'] * 3600 * 24 
x = -data['gps']
dt_std = 60
x_std =1
xSource = 84
ySource =  82
depth_estimate = 3200
ref_point = np.array([-155.274,19.395])  # Ref point is alway CRIM
xy = llh2local(data['LonLat'],ref_point) *1000
radial_distance = ((xy[0] - xSource)**2 + (xy[1] - ySource)**2)**0.5
#radial_distance = 1000
n = np.arange(1,len(tilt_stick)+ 1 )
Nmax = len(n)
nSample = 70000
#Setup inversion
with pm.Model() as model:
    V_exp = pm.Uniform('V_exp',lower = 8,upper = 11)
    k_exp = pm.Uniform('k_exp',lower = 8, upper = 11)
    V = pm.Deterministic('V',10**V_exp)
    k = pm.Deterministic('k',10**k_exp)
    ps = pm.Uniform('ps',lower=1e+6,upper=2e+7)   
    pspd = pm.Uniform('pspd',lower=1e+5,upper=2e+7)
    a = pm.Uniform('a',lower=1,upper=10)
    depth = pm.Normal('depth',mu = depth_estimate, sigma = 500)
    R = (radial_distance**2 + depth**2)**0.5 # Change this if including deepth in the inversion
    R1 = rho * g * V /(k * S)                           #Change this if chainging k or volume

    low_bound = 2 * Nmax * R1 / (1 + R1) * pspd
    up_bound = 2 * (Nmax + 20) * R1 / (1 + R1) * pspd
    plps = pm.Uniform('plps',lower=low_bound,upper=up_bound)
    phi = plps / pspd
    tau =  8 * mu *l * V/ (3.14 * a**4 * k)               #Change this if inverting k or volume
    #Model set-up
    dt_mod = -tau * pm.math.log(((1 + R1) * phi - 2 * n * R1 ) / (((1 + R1) *phi - 2 *(R1*(n -1) -1))))
    x_mod = R1 / (rho * g) * pspd / (1 + R1) * n
    pslip_mod = -ps -rho * g * x_mod
    pstick_mod = pslip_mod + 2 * pspd/ (1 + R1)

#    tilt_mod = (-ps_mod * V_mod  -rho * g * V_mod * x) * tilt_coeff_inversion 
    tilt_coeff =  9./(4*3.14) * V * (1 - poisson)/lame* radial_distance * depth/(R**5) #Change this if inverting volume or depth
    tilt_slip_mod = pslip_mod * tilt_coeff
    tilt_stick_mod =pstick_mod * tilt_coeff
    
    #Posterio
    #Tst_obs = pm.Normal('Tst_obs', mu=tilt_stick_mod, sigma = tilt_std, observed=tilt_stick)
    #Tsl_obs = pm.Normal('Tsl_obs', mu=tilt_slip_mod, sigma = tilt_std, observed=tilt_slip)
    dt_obs = pm.Normal('dt_obs', mu = dt_mod, sigma = 10000, observed = dt_stick)
    x_obs = pm.Normal('x_obs', mu = x_mod, sigma = x_std, observed=x) 
    trace = pm.sample(nSample,init='advi',tune=1000)
map_est = pm.find_MAP(model=model)
depth = map_est['depth']
k = map_est['k']
V = map_est['V']
a = map_est['a']
ps = map_est['ps']
pspd = map_est['pspd']
plps = map_est['plps']
phi = plps / pspd
tau =  8 * mu *l * V/ (3.14 * a**4 * k) 
R = (radial_distance**2 + depth**2)**0.5 # Change this if including deepth in the inversion
R1 = rho * g * V /(k * S)     
dt_stick_mod = -tau * np.log(((1 + R1) * phi - 2 * n * R1 ) / (((1 + R1) *phi - 2 *(R1*(n -1) -1)))) 
x_mod = R1 / (rho * g) * pspd / (1 + R1) * (n - 1)
pslip_mod = -ps -rho * g * x_mod
pstick_mod = pslip_mod + 2 * pspd/ (1 + R1)
tilt_coeff =  9./(4*3.14) * V* (1 - poisson)/lame* radial_distance * depth/(R**5) #Change this if inverting volume or depth
tilt_slip_mod = pslip_mod * tilt_coeff
tilt_stick_mod =pstick_mod * tilt_coeff
fig,ax = plt.subplots(nrows =3,ncols = 1)
ax[0].plot(n,tilt_slip,'ro')
ax[0].plot(n,tilt_slip_mod,'r')
ax[0].plot(n,tilt_stick,'bo')
ax[0].plot(n,tilt_stick_mod,'b')
ax[1].plot(n,x,'go')
ax[1].plot(n,x_mod,'g')
ax[2].plot(n,dt_stick,'go')
ax[2].plot(n,dt_stick_mod,'g')
results = {}
results['MAP'] = map_est
results['trace'] = trace
results['tilt_slip'] = tilt_slip
results['tilt_stick'] = tilt_stick
results['dt_stick'] = tilt_stick
results['x'] = x
results['station'] = tiltName
results['mu'] = mu
results['Rp'] = Rp
results['lame'] = lame
results['nSample'] = nSample
results['radial_distnace'] = radial_distance
pickle.dump(results,open(tiltName + '/' + 'mu_'+ str(mu) + '_Rp_'+ str(Rp)+'.pickle','wb'))