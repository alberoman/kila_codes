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

path_results = '../../../results/'
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
#Setup inversion
pi = 3.14
Niter = 100
conds = 3.5
with pm.Model() as model:
    gpsconst = pm.Uniform('gpsconst',lower = -15,upper = 15)
    A_mod = pm.Uniform('A_mod',lower = 0,upper = 1000)
    B_mod = pm.Uniform('B_mod',lower = 0, upper = 1000)
    E_mod = pm.Uniform('E_mod',lower = 0,upper = 1000)
    
    Vd_exp = pm.Uniform('Vd_exp',lower = 8,upper = 12)
    #kd_exp = pm.Uniform('kd_exp',lower = 8, upper = 11)
    Vs_exp = pm.Uniform('Vs_exp',lower = 8,upper = 11)
    ks_exp = pm.Uniform('ks_exp',lower = 7, upper = 10)
    Vd_mod = pm.Deterministic('Vd_mod',10**Vd_exp)
    #kd_mod = pm.Deterministic('kd_mod',10**kd_exp)
    Vs_mod = pm.Deterministic('Vs_mod',10**Vs_exp)
    #ratio = pm.Uniform('ratio',lower = 0.1,upper = 5e+3)
    ks_mod = pm.Deterministic('ks_mod',10**ks_exp)
    pspd_mod = pm.Uniform('pspd_mod',lower=1e+5,upper=1e+7)
    #conds_mod = pm.Uniform('conds_mod',lower=1,upper=10)
    condd_mod = pm.Uniform('condd_mod',lower=1,upper=30)
    dsh_mod = pm.Normal('dsh_mod',mu = dsh, sigma = dshErr)
    xsh_mod = pm.Normal('xsh_mod',mu = xsh, sigma = xshErr)
    ysh_mod = pm.Normal('ysh_mod',mu = ysh, sigma = yshErr)
    coeffx = cs * dsh_mod * (x -  xsh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vs_mod
    coeffy = cs * dsh_mod * (y -  ysh_mod) / (dsh_mod**2 + (x -  xsh_mod)**2 + (y -  ysh_mod)**2 )**(5./2) * Vs_mod
    R1 = pm.Deterministic('R1',rho * g * Vs_mod /(ks_mod * S)  )
    low_bound = 4 * Nmin * R1 / (1 + R1) 
    up_bound = 4 * Nmax * R1 / (1 + R1) 
    tau2 = 8 * mu *ld * Vs_mod/ (3.14 * condd_mod**4 * ks_mod)    #Model set-up
    x_mod =gpsconst+ 4 * R1 / (rho * g) * pspd_mod / (1 + R1) * n

    
    pslip_mod = -rho * g * x_mod
    pstick_mod = pslip_mod + 4 * pspd_mod/ (1 + R1)

    tslx_mod = coeffx * pslip_mod
    tsly_mod = coeffy * pslip_mod
    tstx_mod = coeffx * pstick_mod
    tsty_mod = coeffy * pstick_mod    

    T1 = (conds_mod / condd_mod )**4 * ld /ls
    phi = 1 * Vs_mod / Vd_mod
    #phi = ratio  * Vs_mod / kd_mod
    stack_mod  = A_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 + tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + B_mod * tt.exp(tstack/tau2*(-T1/2 - phi/2 - tt.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - E_mod
#    stack_mod = A_mod * tt.exp(t*(-condd**4*pi*r/(16*ld*mu) + tt.sqrt((condd**4*pi*r/(8*ld*mu) - condd**4*ks*pi/(8*Vs*ld*mu) - conds**4*ks*pi/(8*Vs*ls*mu))**2 + condd**8*ks*pi**2*r/(16*Vs*ld**2*mu**2))/2 - condd**4*ks*pi/(16*Vs*ld*mu) - conds**4*ks*pi/(16*Vs*ls*mu))) +
#                B_mod * tt.exp
    #Posterio
    tslx_obs = pm.Normal('tslx_obs', mu=tslx_mod, sigma = tilt_std, observed=tiltslx)
    tsly_obs = pm.Normal('tsly_obs', mu=tsly_mod, sigma = tilt_std, observed=tiltsly)
    tstx_obs = pm.Normal('tstx_obs', mu=tstx_mod, sigma = tilt_std, observed=tiltstx)
    tsty_obs = pm.Normal('tsty_obs', mu=tsty_mod, sigma = tilt_std, observed=tiltsty)
    
    
    
    x_obs = pm.Normal('x_obs', mu = x_mod, sigma = gps_std, observed=gps)
    stack_obs = pm.Normal('stack_obs', mu = stack_mod, sigma = tilt_std*1e+6, observed=stack)
    trace2 = pm.load_trace(path_results + 'trace300000_UF')
panda_trace = pm.backends.tracetab.trace_to_dataframe(trace2)
panda_trace['Vs_mod'] = np.log10(panda_trace['Vs_mod']) 
panda_trace['ks_mod'] = np.log10(panda_trace['ks_mod']) 
panda_trace['Vd_mod'] = np.log10(panda_trace['Vd_mod']) 
panda_trace['pspd_mod'] = panda_trace['pspd_mod'] / 1e+6  
filename = 'res300000_UF.pickle'
results =pickle.load(open(path_results + filename,'rb'))

R1 = rho * g * results['MAP']['Vs_mod'] /(results['MAP']['ks_mod'] * S)    
xMAP = results['MAP']['gpsconst']+ 4 * R1 / (rho * g) * results['MAP']['pspd_mod'] / (1 + R1) * n
pslipMAP =  -rho * g * xMAP
pstickMAP = pslipMAP + 4 * results['MAP']['pspd_mod']/ (1 + R1)
T1 = (conds / results['MAP']['condd_mod'] )**4 * ld /ls
phi = 1 * results['MAP']['Vs_mod'] / results['MAP']['Vd_mod']
coeffx = cs * results['MAP']['dsh_mod'] * (x -  results['MAP']['xsh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
coeffy = cs * results['MAP']['dsh_mod'] * (y -  results['MAP']['ysh_mod']) / (results['MAP']['dsh_mod']**2 + (x -  results['MAP']['xsh_mod'])**2 + (y -  results['MAP']['ysh_mod'])**2 )**(5./2) * results['MAP']['Vs_mod']
tau2 = 8 * mu *ld * results['MAP']['Vs_mod']/ (3.14 * results['MAP']['condd_mod']**4 * results['MAP']['ks_mod'])
stackMAP  = results['MAP']['A_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 + np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2)) + results['MAP']['B_mod'] * np.exp(tstack/tau2*(-T1/2 - phi/2 - np.sqrt(4*phi + (-T1 + phi - 1)**2)/2 - 1/2))  - results['MAP']['E_mod']
tslxMAP =coeffx * pslipMAP
tstxMAP =coeffx * pstickMAP
tslyMAP =coeffy * pslipMAP
tstyMAP =coeffy * pstickMAP
#ppc = pm.sample_posterior_predictive(trace2, samples=500, model=model)

xmed = np.median(ppc['x_obs'],axis = 0)
xstd = np.std(ppc['x_obs'],axis = 0)

tslxmed = np.median(ppc['tslx_obs'],axis = 0)
tslymed = np.median(ppc['tsly_obs'],axis = 0)
tslxstd = np.std(ppc['tslx_obs'],axis = 0)
tslystd = np.std(ppc['tsly_obs'],axis = 0)

tstxmed = np.median(ppc['tstx_obs'],axis = 0)
tstymed = np.median(ppc['tsty_obs'],axis = 0)
tstxstd = np.std(ppc['tstx_obs'],axis = 0)
tstystd = np.std(ppc['tsty_obs'],axis = 0)

fig,ax = plt.subplots(nrows = 3 ,ncols = 1)
ax[0].plot(n,gps,'bo')
ax[0].plot(n,xMAP,'ro')
ax[0].errorbar(n,xmed,xstd,fmt='none')
ax[0].set_ylabel('CALM [m]')

ax[1].plot(n,tiltslx*1e+6,'bo')
ax[1].plot(n,tslxMAP*1e+6,'ro')
ax[1].errorbar(n,tslxmed*1e+6,tslxstd*1e+6,fmt='none')

ax[1].plot(n,tiltstx*1e+6,'bo')
ax[1].plot(n,tstxMAP*1e+6,'ro')
ax[1].errorbar(n,tstxmed*1e+6,tstxstd*1e+6,fmt='none')

ax[1].set_ylabel('UWD EW [$\mu$rad]')

ax[2].plot(n,tiltsly*1e+6,'bo')
ax[2].plot(n,tslyMAP*1e+6,'ro')
ax[2].errorbar(n,tslymed*1e+6,tslystd*1e+6,fmt='none')

ax[2].plot(n,tiltsty*1e+6,'bo')
ax[2].plot(n,tstyMAP*1e+6,'ro')
ax[2].errorbar(n,tstymed*1e+6,tstystd*1e+6,fmt='none')

ax[2].set_ylabel('UWD NS [$\mu$rad]')
ax[2].set_xlabel('Collapse number')
plt.tight_layout()

plt.savefig('MAPcollapses.pdf')

stackmed = np.median(ppc['stack_obs'],axis = 0)
stackstd = np.std(ppc['stack_obs'],axis = 0)
plt.figure()
plt.plot(tstack,stack,'b')
plt.plot(tstack,stackMAP,'r')
plt.fill_between(tstack,stackmed-stackstd,stackmed+stackstd,alpha = 0.5)

plt.tight_layout()
plt.savefig('MAPstack.pdf')

#plt.figure()
#corner.corner( panda_trace[['xsh_mod','ysh_mod','dsh_mod','pspd_mod','condd_mod','ks_mod','Vs_mod','Vd_mod']],color = 'black',
#              truths =[results['MAP']['xsh_mod'],results['MAP']['ysh_mod'],results['MAP']['dsh_mod'],results['MAP']['pspd_mod']/1e+6,results['MAP']['condd_mod'],np.log10(results['MAP']['ks_mod']),np.log10(results['MAP']['Vs_mod']),np.log10(results['MAP']['Vd_mod'])])
