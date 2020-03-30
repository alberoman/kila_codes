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

R10 = 0.01
R20 = 1e-6 
R30 = 1.8
R50 = 0.1

#Creating synthetic data
dummy,t_synth = calc_cycles(R10,R20,R30,R50,cycles)
N_cycles = np.ceil((R30 - 1) * (R10 + 1) / (2 * (1 - R50) * R10))
n = np.arange(1, N_cycles)
dt = - 1. /R20 * np.log((R30 * (1+R10) - 2*n*R10*(1 - R50) - (1+R10)) / (R30*(1+R10) - 2 *(1 -R50)*(-1+R10*(n - 1)) - (1+R10)))
Pslip = -1 + 2* R10 * (R50 - 1) / (1 + R10) * (n-1)
Pstick = Pslip + 2*(1-R50)/(1+R10)
Nmax = len(n)

#Adding some noise
sigma_pert_dt = dt[0]/3
np.random.seed(456)
dt_Pert = dt + sigma_pert_dt * np.random.randn(len(dt))
Pslip_Pert = Pslip + 0.3 * np.random.randn(len(dt))
Pstick_Pert = Pstick + 0.3 * np.random.randn(len(dt))

#Setup inversion

with pm.Model() as model:
    R1 = pm.Uniform('R1',lower= 1e-3, upper =0.1 )
    R5 = pm.Uniform('R5',lower= 1e-3, upper =0.99 )

    R2 = pm.Uniform('R2',lower=1e-7, upper=1e-5)
    lowerb =2 * (Nmax)*R1*(1 -R5)/(1 + R1) + 1
    upperb =2 * (Nmax + 10)*R1*(1 -R5)/(1 + R1) + 1
    R3 = pm.Uniform('R3',lower = lowerb ,upper=upperb)

    Pslip_model = -1 + 2* R1 * (R5 - 1) / (1 + R1) * (n-1)
    Pstick_model = -1 + 2* R10 * (R50 - 1) / (1 + R10) * (n-1) + 2*(1-R5)/(1+R1)

    dt_model = - 1. /R2 * np.log((R3 * (1+R1) - 2*n*R1*(1 - R5) - (1+R1)) / (R3*(1+R1) - 2 *(1 -R5)*(-1+R1*(n - 1)) - (1+R1)))
    Psl_obs = pm.Normal('Psl_obs', mu=Pslip_model, sd = 0.1, observed=Pslip_Pert)
    Pst_obs = pm.Normal('Pst_obs', mu=Pstick_model, sd = 0.1, observed=Pstick_Pert)

    dt_obs = pm.Normal('dt_obs', mu=dt_model, sd = 0.5, observed=dt_Pert) 
    step = pm.Metropolis()
    trace = pm.sample(100000,step,chains=2)