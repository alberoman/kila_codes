#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:36:49 2020

@author: aroman
"""


pathfig = 'figs_UWD_offtime/' 
filename = "progress_UWD_offGPS.h5"
backend = emcee.backends.HDFBackend(filename)

np.random.seed(1234)
bndtiltconst = 3000
bndGPSconst = 20
bndtimeconst = 3600 * 24* 20
rho = 2600
g = 9.8
rhog = rho * g
#Conduit parameters
ls = 4e+4
ld = 2.5e+3
mu = 500
#Pressures and friction
pt = 2.0e+7
poisson = 0.25
lame = 1e+9
#Cylinder parameters
Rcyl = 1.0e+3
S = 3.14 * Rcyl**2 
const = -9. /(4 * 3.14) * (1 - poisson) / lame
tiltErr = 1e-5
GPSErr = 1e-2
nstation,x,y,tTilt,tx,ty,tGPS,GPS,locTruth,locErr = pickle.load(open('TiltandGPS_UWD.pickle','rb'))

Nst = 1 + np.max(nstation)
tx = - tx
ty =  -ty
GPS = -GPS
dtx = np.zeros(len(tx))
dty = np.zeros(len(ty))
dtiltErr = 0

Nmax = 40
bounds = np.array([[+9,+11],[+7,+10],[+8,+11],[-2,0],[1.1,10],[1,10],[1,10]]) 
filename = "progress_UWD_offtime.h5"
backend = emcee.backends.HDFBackend(filename)
nwalkers,ndim = backend.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x,y,
                                        ls,ld,pt,mu,
                                        rhog,const,S,
                                        tTilt,tGPS,tx,ty,GPS,
                                        tiltErr,GPSErr,bounds,bndtimeconst,bndGPSconst,Nmax,dtx,dty,dtiltErr,locTruth,locErr,nstation), backend = backend)

samples = sampler.flatchain
parmax = samples[np.argmax(sampler.flatlnprobability)],
offtime,offGPSSamp,offx1,offy1,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp, conddSamp = parmax[0]
offxSamp = np.array([offx1])
offySamp = np.array([offy1])
txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                                              offtime,offGPSSamp,offxSamp,offySamp,xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                              VsExpSamp,VdExpSamp,kExpSamp,R5ExpSamp,R3Samp,condsSamp,conddSamp,
                                              x,y,
                                              ls,ld,pt,mu,
                                              rhog,const,S,nstation)

txmodlist = []
tymodlist = []
GPSmodlist = []
for parameters in samples[np.random.randint(len(samples), size = 100)]:
        #offtimeSamp,offGPSSamp,
        offtime,offGPS,offx1,offy1,xsh,ysh,dsh,xde,yde,dde,Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd = parameters
        offx = np.array([offx1,])
        offy = np.array([offy1,])
        txMod,tyMod,GPSMod = DirectModelEmcee_inv_LF(tTilt,tGPS,
                        offtime,offGPS,offx,offy,xsh,ysh,dsh,xde,yde,dde,
                        Vs_exp,Vd_exp,k_exp,R5exp,R3,conds,condd,
                        x,y,
                        ls,ld,pt,mu,
                        rhog,const,S,nstation)
        
        txmodlist.append(txMod)
        tymodlist.append(tyMod)
        GPSmodlist.append(GPSMod)
        
txspread = np.std(txmodlist,axis =0)
txmed = np.median(txmodlist,axis =0)
tyspread = np.std(tymodlist,axis =0)
tymed = np.median(tymodlist,axis =0)
GPSspread = np.std(GPSmodlist,axis =0)
GPSmed = np.median(GPSmodlist,axis =0)

plt.figure(1)
plt.plot(tTilt / (3600 * 24),tx,'b')
plt.plot(tTilt / (3600 * 24),txMod,'r')
plt.fill_between(tTilt /(3600 * 24),txmed-txspread,txmed + txspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Tilt-x UWD')

plt.savefig(pathfig + 'tx_UWD.pdf')
plt.figure(2)
plt.plot(tTilt / (3600 * 24),ty,'b')
plt.plot(tTilt / (3600 * 24),tyMod,'r')
plt.fill_between(tTilt / (3600 * 24),tymed - tyspread,tymed + tyspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Tilt-y UWD')
plt.savefig(pathfig + 'ty_UWD.pdf')

plt.figure(3)
plt.plot(tGPS /(3600 * 24),GPS,'b')
plt.plot(tGPS / (3600 * 24),GPSMod,'r')
plt.fill_between(tGPS/ (3600 * 24),GPSmed - GPSspread,GPSmed + GPSspread,color='grey',alpha=0.5)
plt.xlabel('Time [Days]')
plt.ylabel('Piston displacement [m]')

plt.savefig(pathfig + 'GPS.pdf')
