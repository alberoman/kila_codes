#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:17:50 2019

@author: aroman
"""
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
from period_vs_n_lib import *
import matplotlib.pyplot as plt

 
    

cycles = 1e+4

g = 9.8
rho = 2700
#chamber and elasticity
V0= 1e+9
radius_cube = 0.75 * V0 / 3.14
#Elastic properties crust
poisson = 0.25
lame = 1e+9
#Position of the chamber
radial_distance = 1000
depth  = 2000
R = (radial_distance**2 + depth**2)**0.5 
k0 = 1.2e+9
#Piston geometrical parameters
Rp = 7e+2
S = 3.14 * Rp**2
#Conduit and mama properties
a = 4.
mu = 5e+2
l = 3e+4
#Friction properties
ps0 = 1e+7
plps0 = 2e+7
pspd0 = 1e+7
phi0 = plps0 / pspd0
R10 = rho * g * V0 /(k0*S)
tau0 =  8 * mu *l * V0 / (3.14 * a**4 * k0)
tilt_coeff =  9./(4*3.14) *V0 * (1 - poisson)/lame* radial_distance * depth/(R**5)

#Creating synthetic data
N_cycles = int(np.ceil((1 + R10) * phi0 / (2*R10)))
n = np.arange(1, N_cycles)
n0 = n -  1
x = R10/(rho * g) * pspd0 / (1 + R10) * n
pslip = -ps0 -rho * g * R10/(rho * g) * pspd0 / (1 + R10) * n0
pstick = pslip + 2*pspd0/(1 + R10)
dt = -tau0 * np.log(((1 + R10)*phi0 - 2 * n * R10 ) / (((1 + R10)*phi0 - 2 *(R10*(n -1) -1))))
tilt_slip =  pslip * tilt_coeff
tilt_stick =  pstick * tilt_coeff

#Adding some noise
pert = 0.3
sigma_dt =  -tau0 * np.log(((1 + R10)*phi0 - 2 * 1 * R10 ) / (((1 + R10)*phi0 - 2 *(-1)))) * pert
sigma_pslip = np.abs(-ps0 -rho * g * R10/(rho * g) * pspd0 / (1 + R10)) * pert
sigma_pstick = np.abs(-ps0 -rho * g * R10/(rho * g) * pspd0 / (1 + R10)) * pert
sigma_x = np.abs(R10/(rho * g) * pspd0 / (1 + R10)) * pert
sigma_tilt = np.abs(sigma_pslip) * tilt_coeff * pert
np.random.seed(456)
dt_pert = dt + sigma_dt * np.random.randn(len(dt))
pslip_pert = pslip + sigma_pslip * np.random.randn(len(dt))
pstick_pert = pstick + sigma_pslip * np.random.randn(len(dt))
x_pert = x + sigma_x * np.random.randn(len(dt))
tilt_slip_pert =  tilt_slip + sigma_tilt * np.random.randn(len(dt))
tilt_stick_pert =  tilt_stick + sigma_tilt * np.random.randn(len(dt))


Nmax = len(n)

#Setup inversion
with pm.Model() as model:
    V_exp = pm.Uniform('V_exp',lower = 8,upper = 11)
    k_exp = pm.Uniform('k_exp',lower = 8, upper = 11)
    V_mod = pm.Deterministic('V_mod',10**V_exp)
    k_mod = pm.Deterministic('k_mod',10**k_exp)
    ps_mod = pm.Uniform('ps_mod',lower=1e+6,upper=3e+7)   
    pspd_mod = pm.Uniform('pspd_mod',lower=1e+6,upper=3e+7)
    a_mod = pm.Uniform('a_mod',lower=1,upper=10)
    depth_mod = pm.Normal('depth_mod',mu = 2200, sigma = 500)
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
    Tst_obs = pm.Normal('Tst_obs', mu=tilt_stick_mod, sigma = sigma_tilt, observed=tilt_stick_pert)
    Tsl_obs = pm.Normal('Tsl_obs', mu=tilt_slip_mod, sigma = sigma_tilt, observed=tilt_slip_pert)
#    pslip_obs = pm.Normal('pslip_obs', mu = pslip_mod, sd = sigma_pslip*10, observed = pslip_pert)
#    pstick_obs = pm.Normal('pstick_obs', mu = pstick_mod, sd = sigma_pstick*10, observed = pstick_pert)
    dt_obs = pm.Normal('dt_obs', mu = dt_mod, sigma = sigma_dt, observed = dt_pert)
    x_obs = pm.Normal('x_obs', mu = x_mod, sigma = sigma_x, observed=x_pert) 
#    tilt_obs = pm.Normal('tilt_obs', mu = tilt_mod, sigma = sigma_tilt, observed = tilt_slip_pert)
    trace = pm.sample(100000,init='advi',tune=1000)
pm.traceplot(trace)
    
    