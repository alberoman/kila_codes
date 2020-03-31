#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:21:02 2020

@author: aroman
plt"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *

xSource = np.array([0,0])
ySource =  np.array([-6000,0])
depthSource = np.array([3000,1000])
xstation1 = -2000
ystation1 = 1000
xstation2 = 0
ystation2 = -1000

rho = 2600
g = 9.8
#Conduit parameters
ls = 3e+4
ld = 2.5e+3
mu = 100

#Pressures and friction6
pt = 20e+6
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 6.0e+2

psData,pdData,tData,psOrig,pdOrig,tx1,ty1,tx2,ty2,tOrig = Twochambers_syntethic_LF_check(xSource,ySource,depthSource,xstation1,xstation2,ystation1,ystation2,ls,ld,mu,pt,Rcyl,rho,g,poisson,lame)
plt.figure(1)
plt.plot(tData,psData,'ro')
plt.plot(tOrig,psOrig,'k')
plt.figure(2)
plt.plot(tData,pdData,'ro')
plt.plot(tOrig,pdOrig,'k')

