#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:46:56 2020

@author: aroman
"""

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


g = 9.8
rho = 2700
#chamber and elasticity
#Elastic properties crust
poisson = 0.25
lame = 3e+9
cs = -9. /(4 * 3.14) * (1 - poisson) / lame
#Position of the chamber
#Piston geometrical parameters
Rp = 7e+2
S = 3.14 * Rp**2
#Conduit Properties
mu = 1e+2
ls = 4e+4
ld = 2.5e+3
data = pickle.load(open('data2ch.pickle','rb'))
dtslip,tiltsty,tiltstx,tiltsly,tiltslx,gps,stack,tstack,xsh,ysh,dsh,xshErr,yshErr,dshErr = data
tilt_std = 1e-5
dt_std = 3600
gps_std = 1
ref_point = np.array([-155.274,19.395])  # Ref point is alway CRIM
x = -1774.
y  = 2867.
Nmin = 2
Nmax = 70
n = np.arange(1,len(tiltsty)+ 1 )

filename = 'res100000_UF.pickle'
results =pickle.load(open(filename,'rb'))

R1 = rho * g * results['MAP']['Vs_mod'] /(results['MAP']['ks_mod'] * S)    
xMAP = results['MAP']['gpsconst']+ 2 * R1 / (rho * g) * results['MAP']['pspd_mod'] / (1 + R1) * n
pslipMAP =  -rho * g * xMAP
pstickMAP = pslipMAP + 2 * results['MAP']['pspd_mod']/ (1 + R1)
T1 = (results['MAP']['conds_mod'] / results['MAP']['condd_mod'] )**4 * ld /ls
phi = results['MAP']['kd_mod'] / results['MAP']['ks_mod'] * results['MAP']['Vs_mod'] / results['MAP']['Vd_mod']
coeffx = cs * results['MAP']['dsh_mod'] * (x -  results['MAP']['xsh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
coeffy = cs * results['MAP']['dsh_mod'] * (y -  results['MAP']['ysh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
tau2 = 8 * mu *ld * results['MAP']['Vs_mod']/ (3.14 * results['MAP']['condd_mod']**4 * results['MAP']['ks_mod'])
stackMAP  = results['MAP']['A_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 + np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + results['MAP']['B_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 - np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - results['MAP']['E_mod']
tslxMAP =coeffx * pslipMAP
tstxMAP =coeffx * pstickMAP
tslyMAP =coeffy * pslipMAP
tstyMAP =coeffy * pstickMAP
fig,ax = plt.subplots(nrows = 3 ,ncols = 1)
ax[0].plot(n,gps,'bo')
ax[0].plot(n,xMAP,'ro')
ax[0].set_ylabel('CALM [m]')
ax[1].plot(n,tiltslx*1e+6,'bo')
ax[1].plot(n,tslxMAP*1e+6,'ro')
ax[1].plot(n,tiltstx*1e+6,'bo')
ax[1].plot(n,tstxMAP*1e+6,'ro')
ax[1].set_ylabel('UWD East [$\mu$rad]')

ax[2].plot(n,tiltsly*1e+6,'bo')
ax[2].plot(n,tslyMAP*1e+6,'ro')
ax[2].plot(n,tiltsty*1e+6,'bo')
ax[2].plot(n,tstyMAP*1e+6,'ro')
ax[2].set_ylabel('UWD North [$\mu$rad]')

ax[2].set_xlabel('Collapse number')
plt.figure(2)
plt.plot(tstack,stack,'b')
plt.plot(tstack,stackMAP,'r')

