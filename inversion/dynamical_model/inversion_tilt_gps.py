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
import pymc3 as pmc
import pickle as pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'../../')
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
mu = 5e+2
l = 4e+4

data = pickle.load(open('DataForInversion.pickle','rb'))
tilt_stick = data['tilt_stick'] * 1e-6
tilt_slip =  data['tilt_slip'] * 1e-6
tilt_std = data['tilt_std'] *1e-6
dt_stick = data['dt_stick'] * 3600 * 24 
x = data['gps']
dt_std = 60
x_std =1
xSource = 84
ySource =  82
depth_estimate = 3200
ref_point = np.array([-155.274,19.395])  # Ref point is alway CRIM
xy = llh2local(data['LonLat'],ref_point)
radial_distance = ((xy[0] - xSource)**2 + (xy[1] - ySource)**2)**0.5
n = np.arange(1,len(tilt_stick)+ 1 )
Nmax = len(n)
#Setup inversion
with pm.Model() as model:
    V_exp = pm.Uniform('V_exp',lower = 8,upper = 11)
    k_exp = pm.Uniform('k_exp',lower = 8, upper = 11)
    V_mod = pm.Deterministic('V_mod',10**V_exp)
    k_mod = pm.Deterministic('k_mod',10**k_exp)
    ps_mod = pm.Uniform('ps_mod',lower=1e+6,upper=3e+7)   
    pspd_mod = pm.Uniform('pspd_mod',lower=1e+5,upper=3e+7)
    a_mod = pm.Uniform('a_mod',lower=1,upper=10)
    depth_mod = pm.Normal('depth_mod',mu = depth_estimate, sigma = 500)
    R = (radial_distance**2 + depth_mod**2)**0.5 # Change this if including deepth in the inversion
    R1 = rho * g * V_mod /(k_mod * S)                           #Change this if chainging k or volume

    low_bound = 2 * Nmax * R1 / (1 + R1) * pspd_mod
    up_bound = 2 * (Nmax + 20) * R1 / (1 + R1) * pspd_mod
    plps_mod = pm.Uniform('plps_mod',lower=low_bound,upper=up_bound)
    phi = plps_mod / pspd_mod
    tau =  8 * mu *l * V_mod/ (3.14 * a_mod**4 * k_mod)               #Change this if inverting k or volume
    #Model set-up
    dt_mod = -tau * pm.math.log(((1 + R1) * phi - 2 * n * R1 ) / (((1 + R1) *phi - 2 *(R1*(n -1) -1))))
    x_mod = R1 / (rho * g) * pspd_mod / (1 + R1) * n
    pslip_mod = -ps_mod -rho * g * x_mod
    pstick_mod = pslip_mod + 2 * pspd_mod/ (1 + R1)

#    tilt_mod = (-ps_mod * V_mod  -rho * g * V_mod * x) * tilt_coeff_inversion 
    tilt_coeff =  9./(4*3.14) * V_mod * (1 - poisson)/lame* radial_distance * depth_mod/(R**5) #Change this if inverting volume or depth
    tilt_slip_mod = pslip_mod * tilt_coeff
    tilt_stick_mod =pstick_mod * tilt_coeff
    
    #Posterio
    Tst_obs = pm.Normal('Tst_obs', mu=tilt_stick_mod, sigma = tilt_std, observed=tilt_stick)
    Tsl_obs = pm.Normal('Tsl_obs', mu=tilt_slip_mod, sigma = tilt_std, observed=tilt_slip)
#    pslip_obs = pm.Normal('pslip_obs', mu = pslip_mod, sd = sigma_pslip*10, observed = pslip_pert)
#    pstick_obs = pm.Normal('pstick_obs', mu = pstick_mod, sd = sigma_pstick*10, observed = pstick_pert)
    dt_obs = pm.Normal('dt_obs', mu = dt_mod, sigma = tilt_std, observed = dt_stick)
    x_obs = pm.Normal('x_obs', mu = x_mod, sigma = x_std, observed=x) 
#    tilt_obs = pm.Normal('tilt_obs', mu = tilt_mod, sigma = sigma_tilt, observed = tilt_slip_pert)
    trace = pm.sample(100000,init='advi',tune=1000)
pm.traceplot(trace)
    
    