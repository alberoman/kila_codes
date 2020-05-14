#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:58:00 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:17:10 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:33:33 2020

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
import corner
sys.path.insert(1,'../../')
import matplotlib.gridspec as gridspec

path_results = '../../../results/'

g = 9.8
rho = 2700
#chamber and elasticity
#Elastic properties crust
poisson = 0.25
lame = 3e+9
lamesar = 1e+9 
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
dtslip,tiltsty,tiltstx,tiltsly,tiltslx,gps,stackx,stacky,tstack,xsh,ysh,dsh,xshErr,yshErr,dshErr,strsrc,strsrcErr = data
stacky = stacky *1e+6
stackx = stackx *1e+6

strsrc = strsrc * lame / lamesar
strsrcErr = strsrcErr * lame / lamesar
tilt_std = 1e-5
dt_std = 3600
gps_std = 1
ref_point = np.array([-155.274,19.395])  # Ref point is alway CRIM
x = -1774.
y  = 2867.
Nmin = 2
Nmax = 70
n = np.arange(1,len(tiltsty)+ 1 )
#Setup inversion
pi = 3.14
Niter = 300000
#conds_mod = 3.5
path_results = '../../../results/'
 
filename = 'res300000_strsrc_UF.pickle'
results =pickle.load(open(path_results + filename,'rb'))

R1 = rho * g * results['MAP']['Vs_mod'] /(results['MAP']['ks_mod'] * S)    
xMAP = results['MAP']['gpsconst']+ 4 * R1 / (rho * g) * results['MAP']['pspd_mod'] / (1 + R1) * n
pslipMAP =  -rho * g * xMAP
pstickMAP = pslipMAP + 4 * results['MAP']['pspd_mod']/ (1 + R1)
T1 = (results['MAP']['conds_mod']/ results['MAP']['condd_mod'] )**4 * ld /ls
phi = 1 * results['MAP']['Vs_mod'] / results['MAP']['Vd_mod']
coeffx = cs * results['MAP']['dsh_mod'] * (x -  results['MAP']['xsh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
coeffy = cs * results['MAP']['dsh_mod'] * (y -  results['MAP']['ysh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
tau2 = 8 * mu *ld * results['MAP']['Vs_mod']/ (3.14 * results['MAP']['condd_mod']**4 * results['MAP']['ks_mod'])
#stackMAPy  = -(results['MAP']['A_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 + np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + results['MAP']['B_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 - np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - results['MAP']['E_mod'])
stackMAPx  = coeffx *1e+6*results['MAP']['pspd_mod']*(results['MAP']['C_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 + np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + results['MAP']['D_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 - np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - results['MAP']['F_mod'])

tslxMAP =coeffx * pslipMAP
tstxMAP =coeffx * pstickMAP
tslyMAP =coeffy * pslipMAP
tstyMAP =coeffy * pstickMAP






plt.figure()
plt.plot(tstack,stacky,'b')
#plt.plot(tstack,stackMAPy,'m')
ax = plt.gca()
ax.set_xlabel('Time [Hours]',fontsize= 12)
ax.set_ylabel('UWD NS [$\mu rad$]',fontsize = 12)
#ax.set_xlim([0,24])
plt.figure()
plt.plot(tstack,stackx,'b')
plt.plot(tstack,stackMAPx,'m')
ax = plt.gca()
ax.set_xlabel('Time [Hours]',fontsize= 12)
ax.set_ylabel('UWD EW [$\mu rad$]',fontsize = 12)
plt.savefig('MAPstack_strsrc_UF.pdf')

