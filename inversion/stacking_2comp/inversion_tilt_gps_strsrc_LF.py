
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
sys.path.insert(1,'../../')


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

strsrc = np.abs(strsrc)
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
path_results = '../../../results/'
with pm.Model() as model:
    gpsconst = pm.Uniform('gpsconst',lower = -15,upper = 15)
    A_mod = pm.Uniform('A_mod',lower = 0, upper = 1e+2)
    B_mod = pm.Uniform('B_mod',lower = 0, upper = 1e+2)
    C_mod = pm.Uniform('C_mod',lower = 0, upper = 1e+2)
    D_mod = pm.Uniform('D_mod',lower = 0, upper = 1e+2)
    E_mod = pm.Uniform('E_mod',lower = 0, upper = 1e+2)
    F_mod = pm.Uniform('F_mod',lower = 0, upper = 1e+2)

    Vd_exp = pm.Uniform('Vd_exp',lower = 8,upper = 11)
    Vd_mod = pm.Deterministic('Vd_mod',10**Vd_exp)

    kd_exp = pm.Uniform('kd_exp',lower = 7, upper = 10)
    kd_mod = pm.Deterministic('kd_mod',10**kd_exp)
    R1 = pm.Deterministic('R1',rho * g * Vd_mod /(kd_mod * S)  )
    ratio = pm.Uniform('ratio',lower = 30 * 4 * R1 / (1 + R1), upper = 100 * 4 * R1 / (1 + R1))
    pspd_mod = pm.Uniform('pspd_mod',lower=1e+5,upper=1e+7)
    ptps_mod = pm.Deterministic('ptps_mod',ratio * pspd_mod)
    conds_mod = pm.Uniform('conds_mod',lower = 1, upper = 10)

    deltap_mod = pm.Uniform('deltap',lower = 1e+5,upper = ptps_mod )
    strsrc_mod = pm.Normal('strsrc_Mod',mu = strsrc, sigma = strsrcErr)
    Vs_mod = pm.Deterministic('Vs_mod',strsrc_mod / deltap_mod)

    
    #conds_mod = pm.Uniform('conds_mod',lower=1,upper=10)
    condd_mod = pm.Uniform('condd_mod',lower=1,upper=30)
    dsh_mod = pm.Normal('dsh_mod',mu = dsh, sigma = dshErr)
    xsh_mod = pm.Normal('xsh_mod',mu = xsh, sigma = xshErr)
    ysh_mod = pm.Normal('ysh_mod',mu = ysh, sigma = yshErr)
    coeffx = cs * dsh_mod * (x -  xsh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vd_mod
    coeffy = cs * dsh_mod * (y -  ysh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vd_mod
    tau2 = 8 * mu *ld * Vs_mod/ (3.14 * condd_mod**4 * kd_mod)    #Model set-up
    x_mod =gpsconst+ 4 * R1 / (rho * g) * pspd_mod / (1 + R1) * n
    
    pslip_mod = -rho * g * x_mod
    pstick_mod = pslip_mod + 4 * pspd_mod/ (1 + R1)

    tslx_mod = coeffx * pslip_mod
    tsly_mod = coeffy * pslip_mod
    tstx_mod = coeffx * pstick_mod
    tsty_mod = coeffy * pstick_mod    

    T1 = (conds_mod / condd_mod )**4 * ld /ls
    phi = 1 * Vs_mod / Vd_mod
    stacky_mod  =   coeffy * 1e+6* pspd_mod *(A_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 + tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + B_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 - tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - ratio)
    #stackx_mod  =   coeffx * 1e+6* pspd_mod *(C_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 + tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + D_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 - tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - ratio)

    tslx_obs = pm.Normal('tslx_obs', mu=tslx_mod, sigma = tilt_std, observed=tiltslx)
    tsly_obs = pm.Normal('tsly_obs', mu=tsly_mod, sigma = tilt_std, observed=tiltsly)
    tstx_obs = pm.Normal('tstx_obs', mu=tstx_mod, sigma = tilt_std, observed=tiltstx)
    tsty_obs = pm.Normal('tsty_obs', mu=tsty_mod, sigma = tilt_std, observed=tiltsty)
    
    x_obs = pm.Normal('x_obs', mu = x_mod, sigma = gps_std, observed=gps)
    #stackx_obs = pm.Normal('stackx_obs', mu = stackx_mod, sigma = tilt_std*1e+6, observed=stackx)
    stacky_obs = pm.Normal('stacky_obs', mu = stacky_mod, sigma = tilt_std*1e+6, observed=stacky)

    trace = pm.sample(Niter,init='advi',tune=100,target_accept =0.85)
map_estimate = pm.find_MAP(model=model )
results = {}
results['MAP'] = map_estimate
results['iterations'] = Niter
pickle.dump(results,open(path_results + 'res' + str(Niter) + '_strsrc_LF.pickle','wb'))
pm.save_trace(trace, path_results +'trace'+str(Niter) + '_strsrc_LF',overwrite =  True)
#pm.traceplot(trace)
    
 