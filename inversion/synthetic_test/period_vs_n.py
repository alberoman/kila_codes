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
k = 1.2e+9
#Piston geometrical parameters
Rp = 7e+2
hp = 5e+2
S = 3.14 * Rp**2
Vp = S * hp
m = rho * Vp
#Conduit and mama properties
a = 4.
mu = 5e+2
l = 3e+4
#Friction properties
c0 = 10e+6
Fs = c0 * S
R50 = 0.1
Fd = Fs / R50
#Driving pressure
Pl =2.0e+7
tstar = ((m*V0) / (k* S**2))**0.5
xstar = Fs*V0/(k*S**2)
pstar = Fs/S
R10 = rho * g * V0 /(k*S)
R20 = (m * k / V0)**0.5 *(3.14 * a**4 )/(8* mu * l* S) 
R30 = Pl / Fs * S

#Creating synthetic data
dummy,t_synth = calc_cycles(R10,R20,R30,R50,cycles)
N_cycles = np.ceil((R30 - 1) * (R10 + 1) / (2 * (1 - R50) * R10))
n = np.arange(1, N_cycles)
dt = - 1. /R20 * np.log((R30 * (1+R10) - 2*n*R10*(1 - R50) - (1+R10)) / (R30*(1+R10) - 2 *(1 -R50)*(-1+R10*(n - 1)) - (1+R10)))
Pslip = -1 - 2* R10 * (n -1)* (1 + R50) / (1 + R10)
Pstick = Pslip + 2*(1-R50)/(1+R10)
x = 2 * n * (1- R50)/(1 + R10)
#Upscale
dt = dt *tstar
Pslip = Pslip *pstar
Pstick = Pstick*pstar
x = x * xstar
#Adding some noise
sigma_pert_dt = dt[0]/10 
sigma_pert_Pslip = np.abs(Pslip[0])/10
sigma_pert_x = 2* (1-R50)/(1 +R10) * xstar/10
sigma_pert_tilt = 9./(4*3.14) * V0 * sigma_pert_Pslip * (1 - poisson)/lame* radial_distance * depth/(R**5) 

np.random.seed(456)
dt_Pert = dt + sigma_pert_dt * np.random.randn(len(dt))
Pslip_Pert = Pslip + sigma_pert_Pslip * np.random.randn(len(dt))
Pstick_Pert = Pstick + sigma_pert_Pslip * np.random.randn(len(dt))
x_Pert = x + sigma_pert_x * np.random.randn(len(dt))

#Calculating tilt
tilt_slip_Pert = 9./(4*3.14) * V0 * Pslip_Pert * (1 - poisson)/lame* radial_distance * depth/(R**5)
tilt_stick_Pert = 9./(4*3.14) * V0 * Pstick_Pert * (1 - poisson)/lame* radial_distance * depth/(R**5)
Nmax = len(n)


#Setup inversion
with pm.Model() as model:
#    Pstick_Pert = Pstick + sigma_pert_Pslip * np.random.randn(len(dt))
    
    V_exp = pm.Uniform('V_exp',lower = 8,upper = 10)
    V = pm.Deterministic('V',10**V_exp)
    comp_exp = pm.Uniform('comp_exp',lower = 8,upper = 11)
    comp = pm.Deterministic('comp',10**comp_exp)
#    height = pm.Uniform('height',lower = 100,upper = 1000)
#    mass = rho * S * height
    cond_radius = pm.Uniform('cond_radius',lower = 2, upper = 7)
    BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
    viscosity = BoundedNormal('viscosity', mu= 400, sigma = 200)
    R1 = pm.Deterministic('R1',rho *g * V/ (S * comp))
    R5 = pm.Uniform('R5',lower= 1e-3, upper =0.8 )
    lowerb =2 * (Nmax)*R1*(1 -R5)/(1 + R1) + 1
    upperb =2 * (Nmax + 10)*R1*(1 -R5)/(1 + R1) + 1
    R3 = pm.Uniform('R3',lower = lowerb ,upper=upperb)
    R2 = pm.Deterministic('R2',(m * comp / V)**0.5 *(3.14 * cond_radius**4 )/(8 * viscosity * l* S))
    
    #Model set-up
    Pslip_model = -1 - 2* R1 * (n -1)* (1 +R5) / (1 + R1)
    Pstick_model = -1 - 2* R1 * (n -1)* (1 +R5) / (1 + R1)  + 2*(1-R5)/(1+R1)
    dt_model =- 1. /R2 * tt.log((R3 * (1+R1) - 2*n*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*(n - 1)) - (1+R1)))
    x_model = 2 * n * (1- R5)/(1 + R1)
    #Lengthscales
    pst = pm.Uniform('pst',lower = 1e+6,upper = 3e+7)
    tst = ((m*V) / (comp * S**2))**0.5
    Fst = pst*S
    xst = Fst*V/(k*S**2)
    #Upscale
    dt_model_dimensional = dt_model * tst 
    Pslip_model_dimensional = Pslip_model  * pst
    Pstick_model_dimensional = Pstick_model * pst
    Pl = pm.Deterministic('Pl', R3 * pst)
    x_model_dimensional = x_model * xst
    tilt_stick_model = 9./(4*3.14) * V * Pstick_model_dimensional * (1 - poisson)/lame* radial_distance * depth/(R**5)
    tilt_slip_model = 9./(4*3.14) * V * Pslip_model_dimensional * (1 - poisson)/lame* radial_distance * depth/(R**5)
    #Posterior
  #  Tst_obs = pm.Normal('Tst_obs', mu=tilt_stick_model, sigma = sigma_pert_tilt, observed=tilt_stick_Pert)
  #  Tsl_obs = pm.Normal('Tsl_obs', mu=tilt_slip_model, sigma = sigma_pert_tilt, observed=tilt_slip_Pert)
    Psl_obs = pm.Normal('Psl_obs', mu=Pslip_model_dimensional, sd = sigma_pert_Pslip, observed=Pslip_Pert)
    Pst_obs = pm.Normal('Pst_obs', mu=Pstick_model_dimensional, sd = sigma_pert_Pslip, observed=Pstick_Pert)
    dt_obs = pm.Normal('dt_obs', mu=dt_model_dimensional, sigma = sigma_pert_dt, observed=dt_Pert)
    x_obs = pm.Normal('x_obs', mu=x_model_dimensional, sigma = sigma_pert_x, observed=x_Pert) 
    
    step = pm.Metropolis()
    trace = pm.sample(100000,step,chains=2)