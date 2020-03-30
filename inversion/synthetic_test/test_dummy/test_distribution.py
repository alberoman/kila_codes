#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:08:13 2019

@author: aroman
"""

import numpy as np
import theano.tensor as tt
import pymc3 as pm
import matplotlib.pyplot as plt
x = np.linspace(0,10)
a0= 4
b0 = 3
y = a0 * b0 + b0 * x 
y_Pert= y + 3 * np.random.randn(len(y))

with pm.Model() as model:
    a = pm.Uniform('a',lower = 0, upper = 20)
    b = pm.Uniform('b',lower = 0, upper = 20)
    y_mod = a * b  + b * x

    y_obs = pm.Normal('y_obs', mu=y_mod, sigma = 3, observed = y_Pert)

    step = pm.Metropolis()
    trace = pm.sample(20000,step,chains=2)
