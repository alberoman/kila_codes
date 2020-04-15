#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
plt"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm

def pressure_inversion(t_tilt,t_gps,R1,R3,R5,T1,PHI):
    #Calcualte the  pressure in the two chambers and the 
    un_plus_T1 = 1 + T1
    un_plus_R1 = 1 + R1
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1,R3,R5,T1,PHI,A,B,C,D]
    
    ps_data = np.ones(len(t_tilt))
    pd_data = np.ones(len(t_tilt))
    gps_data = np.ones(len(t_gps))
    PD0 =  - 1  + 2 * (1 - R5) / un_plus_R1 
    PSLIP = - 1 - 2 * R1 * (1 - R5)/un_plus_R1
    PS0 = -1
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1))) - 1
    for i in range(1,N_cycles + 1):
        tslip,ps_data,pd_data,ps0,gps_data = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,t_tilt,ps_data,pd_data,t_gps,gps_data,i)
        PD0 = - 1  + 2 * (1 - R5) / un_plus_R1 -2 * R1 * (1 - R5)/un_plus_R1 * i
        PS0 = ps0
        PSLIP =  - 1 - 2 * R1 * (1 - R5)/un_plus_R1 * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
    
    return ps_data,pd_data,gps_data
    
xSource = np.array([0,0]) #The first value here corresponds to the shallow source (the one feeding)
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = 100
ystation2 = -1000
sigma_tilt = 0.1
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 3e+4
ld = 2.5e+3
condd = 9
conds = 4
#Sources volumes and compressibilities
Vs = 1e+10
Vd  = 5e+9
ks = 1e+9
kd = 1e+9
#Pressures and friction6
taus = 5e+6
taud = 3e+6
mu = 100

#Pressures and friction6
pt = 20e+6
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
tiltPert = 0.001
cs = -9. /(4 * 3.14) * (1 - poisson) / lame
coeffxs1 = cs * depthSource[0] * (xstation1 -  xSource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffys1 = cs * depthSource[0] * (ystation1 -  ySource[0]) / (depthSource[0]**2 + (xstation1 -  xSource[0])**2 + (ystation1 -  ySource[0])**2 )**(5./2)
coeffxd1 = cs * depthSource[1] * (xstation1 -  xSource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)
coeffyd1 = cs * depthSource[1] * (ystation1 -  ySource[1]) / (depthSource[1]**2 + (xstation1 -  xSource[1])**2 + (ystation1 -  ySource[1])**2 )**(5./2)

coeffxs2 = cs * depthSource[0] * (xstation2 -  xSource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffys2 = cs * depthSource[0] * (ystation2 -  ySource[0]) / (depthSource[0]**2 + (xstation2 -  xSource[0])**2 + (ystation2 -  ySource[0])**2 )**(5./2)
coeffxd2 = cs * depthSource[1] * (xstation2 -  xSource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)
coeffyd2 = cs * depthSource[1] * (ystation2 -  ySource[1]) / (depthSource[1]**2 + (xstation2 -  xSource[1])**2 + (ystation2 -  ySource[1])**2 )**(5./2)


tData,tx1,ty1,tx2,ty2,GPS,tGPS,Nmax = Twochambers_syntethic_LF(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,conds,condd,Vs,Vd,ks,kd,taus,taud,mu,pt,Rcyl,rho,g,poisson,lame,tiltPert)
tData_new = np.linspace(1,1e+7,1000000)

Vs_mod = Vs
Vd_mod = Vd
ks_mod = ks
kd_mod = kd
taus_mod = taus   
taud_mod = taud
conds_mod = conds
condd_mod =condd
r1 = rhog * Vd_mod /(kd_mod*S)
pt_mod = pt
t1 = (conds_mod / condd_mod )**4 * ld /ls
phi = kd_mod /ks_mod * Vs_mod / Vd_mod
r5 = taud_mod / taus_mod
r3 = pt_mod  / taus_mod
tstar = Vs_mod * 8 * mu * ld / (ks_mod * 3.14 * condd_mod**4)
xstar = taus_mod * Vd_mod / (kd_mod * S) 
tadim_tilt = tData_new / tstar
tadim_gps  =tGPS /tstar
ps_mod,pd_mod,gps_mod = pressure_inversion(tadim_tilt,tadim_gps,r1,r3,r5,t1,phi)
ps_mod = ps_mod * taus_mod
pd_mod = pd_mod * taus_mod
gps_mod = gps_mod * xstar    
tx1_mod = coeffxs1 * Vs_mod * ps_mod + coeffxd1 * Vd_mod * pd_mod
ty1_mod = coeffys1 * Vs_mod * ps_mod + coeffyd1 * Vd_mod * pd_mod
tx2_mod = coeffxs2 * Vs_mod * ps_mod + coeffxd2 * Vd_mod * pd_mod
ty2_mod = coeffys2 * Vs_mod * ps_mod + coeffyd2 * Vd_mod * pd_mod
plt.plot(tData,tx1,'r',tData_new,tx1_mod,'b'),plt.show()

