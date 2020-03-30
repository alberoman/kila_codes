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
pathgg_results = 'mu_100_lame_1e9_Rp_700/'
filename = 'InversionResultsUWD.pickle'
rho = 2600
g = 9.8
lame = 1e+9
poisson = 0.25
mu = 100
l = 4e+4

results =pickle.load(open(pathgg_results + filename,'rb'))
panda_trace = pm.backends.tracetab.trace_to_dataframe(results['trace'])
panda_trace['V'] = np.log10(panda_trace['V']) 
panda_trace['k'] = np.log10(panda_trace['k']) 
panda_trace['ps'] = panda_trace['ps'] / 1e+6 
panda_trace['pspd'] = panda_trace['pspd'] / 1e+6  
panda_trace['plps'] = panda_trace['plps'] / 1e+6                        #Change this if chainging k or volume
plt.figure(1)
corner.corner( panda_trace[['pspd','plps','ps','a','k','V']],color = 'yellow',
              truths = [results['MAP']['pspd']/1e+6,results['MAP']['plps']/1e+6,results['MAP']['ps']/1e+6,
                        results['MAP']['a'],np.log10(results['MAP']['k']),np.log10(results['MAP']['V'])])
