#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:36:15 2020

@author: aroman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:43:18 2020

@author: aroman
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from gamma_lib import *
import os
import corner
from shutil import copyfile
discardval = 30000
thinval = 1

path_results = '../../../../results/gamma/'
pathrun = 'sigma'
model_type = 'LF'

stations  = ['UWD','SDH','IKI']
date = '07-03-2018'
flaglocation = 'N'      # This is the flag for locations priors, F for Uniform, N for Normal

if len(stations) > 1:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + str(len(stations)) + 'st'
else:
    pathtrunk = 'priors' + flaglocation +  '/' + model_type + '/' + stations[0]



pathtrunk = pathtrunk + '/' + date + '/'

pathgg = pathtrunk + pathrun
pathgg  =  pathgg + '/'
pathgg = path_results + pathgg

copyfile(pathgg + 'progress.h5', pathgg + 'progress_temp.h5' )
pathfig = pathgg + 'figs/'

try:
    os.mkdir(pathfig)
except:
    print('Directory already exist')
#parameters = pickle.load(open(pathgg + 'parameters.pickle','rb')) 
#fix_par = parameters['fix_par']
##tx = parameters['tx']
#ty = parameters['ty']
#dtx = np.zeros(np.shape(tx))
#dty = np.zeros(np.shape(ty))

#GPS = parameters['GPS']
#x,y,ls,ld,pt,mu,rhog,const,S,Nmax,tiltErr,GPSErr = fix_par

np.random.seed(1234)

Nst,nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr,t0= pickle.load(open(pathgg + 'data.pickle','rb'))
a = pickle.load(open(pathgg + 'parameters.pickle','rb'))
if len(a)== 17:
    bounds,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
    boundsLoc = 0
elif len(a)== 18:
    bounds,boundsLoc,bndtiltconst,bndGPSconst,tiltErr,GPSErr,bndp0,locErrFact,a_parameter,thin,nwalkers,ls,ld,mu,ndim,const,S,rhog = a
filename = 'progress_temp.h5'
#Getting sampler info
#nwalkers,ndim = reader.shape
#backend = emcee.backends.HDFBackend(pathgg + filename)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_UF,
#                                   args=(x,y,
#                                            ls,ld,mu,
#                                            rhog,const,S,
#                                            tTilt,tGPS,tx,ty,GPS,
#                                            tiltErr,GPSErr,bounds,bndGPSconst,bndtiltconst,bndp0,locTruth,locErr,nstation), backend = backend)

reader = emcee.backends.HDFBackend(pathgg + filename, read_only = True )
nwalkers,ndim = reader.shape
samples = reader.get_chain(flat = True,discard = discardval)
parmax = samples[np.argmax(reader.get_log_prob(flat = True,discard = discardval))]
if Nst == 1:
    deltap0Samp,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp,alphaSamp, gammaSamp = parmax
    offxSamp = np.array([offx1])
    offySamp = np.array([offy1])
elif Nst == 2:
    deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp,alphaSamp,gammaSamp = parmax
    offxSamp = np.array([offx1],offx2)
    offySamp = np.array([offy1,offy2])
elif Nst == 3:
    deltap0Samp,offGPSSamp,offx1,offy1,offx2,offy2,offx3,offy3,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp,alphaSamp,gammaSamp = parmax
    offxSamp = np.array([offx1,offx2,offx3])
    offySamp = np.array([offy1,offy2,offy3])    
if model_type == 'UF':
    txModbest,tyModbest,GPSModbest = DirectModelEmcee_inv_UF(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,alphaSamp,gammaSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
elif model_type == 'LF':
    txModbest,tyModbest,GPSModbest,ps,pd,coeffxs,coeffxd,coeffys,coeffyd = DirectModelEmcee_inv_LF_diagno(tTilt,tGPS,
                                              deltap0Samp,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,ksExpSamp,kdExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,alphaSamp,gammaSamp,
                                              x,y,
                                              ls,ld,mu,
                                              rhog,const,S,nstation)
plt.figure()
txd = coeffxd[nstation==0] * 10** VdExpSamp * pd[nstation == 0]
txs = coeffxs[nstation==0] * 10** VsExpSamp * ps[nstation == 0]
plt.plot(tTilt[nstation==0],txd,'b')
plt.plot(tTilt[nstation==0],txs,'r')
plt.legend(['HLM','SCR'])

plt.title('UWD-x')
plt.figure()
txd = coeffxd[nstation==1] * 10** VdExpSamp * pd[nstation == 1]
txs = coeffxs[nstation==1] * 10** VsExpSamp * ps[nstation == 1]
plt.plot(tTilt[nstation==1],txd,'b')
plt.plot(tTilt[nstation==1],txs,'r')
plt.legend(['HLM','SCR'])

plt.title('SDH-x')

plt.figure()
tyd = coeffyd[nstation==0] * 10** VdExpSamp * pd[nstation == 0]
tys = coeffys[nstation==0] * 10** VsExpSamp * ps[nstation == 0]
plt.plot(tTilt[nstation==0],tyd,'b')
plt.plot(tTilt[nstation==0],tys,'r')
plt.title('UWD-y')
plt.legend(['HLM','SCR'])

plt.figure()
tyd = coeffyd[nstation==1] * 10** VdExpSamp * pd[nstation == 1]
tys = coeffys[nstation==1] * 10** VsExpSamp * ps[nstation == 1]
plt.plot(tTilt[nstation==1],tyd,'b')
plt.plot(tTilt[nstation==1],tys,'r')
plt.plot(tTilt[nstation==1],tys + 2.5*tyd,'g')
plt.title('SDH-y')
plt.legend(['HLM','SCR','sum'])
