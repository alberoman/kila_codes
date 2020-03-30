#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:13:43 
Make a movie of a two source model with piston collapse
@author: aroman
"""
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar # For dealing with Colorbars the proper way - TBD in a separate PyCoffee ?

import numpy as np
def draw_vector(v0, v1,cl, ax ,head =0):
    #Draws a vector with no arrow using annotation. v0 and v1 are two vectors containing the coordinates of the starting point of the arrow (v0)
    #and the final point of the arrow (v1)
    if head:
        arrowprops=dict(arrowstyle = '->, head_width=0.3, head_length=0.5',color= cl,
                    linewidth=1,
                    shrinkA=0, shrinkB=0)
    else:
        arrowprops=dict(arrowstyle='-',color= cl,
                    linewidth=1,
                    shrinkA=0, shrinkB=0)
    h = ax.annotate('', v1, v0, arrowprops=arrowprops)
    return h
xSource = np.array([0,0])
ySource =  np.array([0,-3000])
depthSource = np.array([1500,3000])
xstation1 = 80
ystation1 = 120
xstation2 = 80
ystation2 = 70

condd = 1.5
conds = 4
ls = 3e+4
ld = 2.5e+3
Vs = 2e+9
Vd  = 8e+9
ks = 1e+9
kd = 1e+9
pd0 = 0e+6
pd0_lin = 0e+7
pt = 15e+6
mu = 100

taus = 5e+6
taud = 1e+6
rho = 2600

g = 9.8
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 
R1 = rho * g * Vs /(ks*S)
T1 = (conds / condd )**4 * ld /ls
un_plus_T1 = 1 + T1
un_plus_R1 = 1 + R1
ps0 = - taus  + 2 * (taus - taud) / (1 + R1)
phi = kd /ks * Vs / Vd
tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
ps_analytic = []
pd_analytic = []
pdslip = []
time =[]
n_cycle = 1
ps0 = - taus  + 2 * (taus - taud) / un_plus_R1 * (1 + (1 - n_cycle) * R1)
numerator = -taus - 2 * R1 * n_cycle * (taus - taud) / un_plus_R1 - pd0 / un_plus_T1  + pt * T1 / un_plus_T1
denominator = (ps0 - pd0 * 1 / un_plus_T1 + pt * T1 / un_plus_T1)
tend = - tstar / un_plus_T1 * np.log(numerator/denominator)

t0 = 0
tend = 0
while not np.isnan(tend):
#while n < 3:
    print(n_cycle)
    t = np.linspace(0,tend,15)
    ps0 = - taus  + 2 * (taus - taud) / un_plus_R1 * (1 + (1 - n_cycle) * R1)
    ps = pd0 * 1 / un_plus_T1 - pt * T1 / un_plus_T1 + (ps0 - pd0 * 1 / un_plus_T1 + pt * T1 / un_plus_T1) * np.exp( - un_plus_T1 * t/tstar )
    pd = pd0 + kd * 3.14 * condd**4 / (Vd * 8 * mu * ld)*(-pd0 + pd0 / un_plus_T1 - pt * T1 / un_plus_T1) * t  
    + phi / un_plus_T1 * (ps0 - pd0 / un_plus_T1 + pt * T1 / un_plus_T1) * (np.exp(- un_plus_T1 * t /tstar) - 1)
    t = t + t0
    pd0 = pd[-1]
    t0 = t[-1] 
    numerator = -taus - 2 * R1 * n_cycle * (taus - taud) / un_plus_R1 - pd0 / un_plus_T1  + pt * T1 / un_plus_T1
    denominator = (ps0 - pd0 * 1 / un_plus_T1 + pt * T1 / un_plus_T1)
    tend = - tstar / un_plus_T1 * np.log(numerator/denominator)

    ps_analytic.append(ps)
    pd_analytic.append(pd)
    pdslip.append(pd[0])
    time.append(t)
    n_cycle = n_cycle + 1
time = np.concatenate(time,axis = 0)

ps = np.concatenate(ps_analytic,axis = 0)
ps = np.reshape(ps,(len(ps),1))
pd = np.concatenate(pd_analytic,axis = 0)
pd = np.reshape(pd,(len(pd),1))
pdslip = np.array(pdslip)
pressure = np.concatenate((ps - ps[0],pd - pd [0]),axis = 1)
V = np.array([Vs,Vd])
poisson = 0.25
lame = 1e+9
x = np.linspace(-10000,10000,200)
y = np.linspace(-10000,10000,200)
X,Y = np.meshgrid(x,y)

V = np.array([Vs,Vd])
deltaP = np.concatenate((ps,pd),axis = 1)
radial_distance = np.zeros((2,np.shape(X)[0],np.shape(X)[1]))
radial_distance[0,:,:] = np.sqrt((X - xSource[0])**2 + (Y - ySource[0])**2)
radial_distance[1,:,:] = np.sqrt((X - xSource[1])**2 + (Y - ySource[1])**2)
uzMax = np.array([3./(3.14* 4) * V[0] * (1 - poisson) / lame * deltaP[:,0] / depthSource[0]**2, 3./(3.14* 4) * V[1] * (1 - poisson) / lame * deltaP[:,1]/ depthSource[1]**2] )
uz = np.zeros((len(time),2,np.shape(radial_distance)[1],np.shape(radial_distance)[2]))

for i in np.arange(len(time)):
    uz[i,0,:,:] = uzMax[0,i] * ( 1 + (radial_distance[0,:,:] / depthSource[0])**2)**(- 3./2)
    uz[i,1,:,:] = uzMax[1,i] * ( 1 + (radial_distance[1,:,:] / depthSource[1])**2)**(- 3./2)
uz[:,0,:,:] = uz[:,0,:,:] - uz[0,0,:,:]
uz[:,1,:,:] = uz[:,1,:,:] - uz[0,1,:,:]
uztot = uz[:,0,:,:] + uz[:,1,:,:]



cordx = np.array([x[xstation1],x[xstation2]])
cordy = np.array([y[ystation1],y[ystation2]])

tilt = np.zeros((len(cordx),len(time),2)) 
cs = -9. /(4 * 3.14) * (1 - poisson) / lame   
 

for i in np.arange(len(cordx)):
    dummy_tiltx = np.zeros((len(time)))
    dummy_tilty = np.zeros((len(time)))
    for jj in np.arange(len(xSource)):
        tempx = cs * V[jj] * pressure[:,jj] * depthSource[jj] * (cordx[i] -  xSource[jj])  / (depthSource[jj]**2 + (cordx[i] -  xSource[jj])**2 + (cordy[i] -  ySource[jj])**2 )**(5./2)
        tempy = cs * V[jj] * pressure[:,jj] * depthSource[jj] * (cordy[i] -  ySource[jj])  / (depthSource[jj]**2 + (cordx[i] -  xSource[jj])**2 + (cordy[i] -  ySource[jj])**2 )**(5./2)
        dummy_tiltx = dummy_tiltx + tempx
        dummy_tilty = dummy_tilty + tempy
    tilt[i,:,0] = dummy_tiltx[:] - dummy_tiltx[0]
    tilt[i,:,1] = dummy_tilty[:] - dummy_tilty[0]
magnitude1 = -(tilt[0,:,0]**2 + tilt[0,:,1]**2)**0.5
magnitude2 = -(tilt[1,:,0]**2 + tilt[1,:,1]**2)**0.5
npix = 1
time_tilt =time / (3600 * 24)
X = X /1e+3
Y = Y / 1e+3
cordx = cordx /1e+3
cordy = cordy /1e+3
filename_counter = 0
for i in np.arange(len(magnitude1)):
    
    fig = plt.figure(figsize =(4.1,4.1))
    gs = fig.add_gridspec(28,28)
    
    ax_cb = plt.subplot(gs[0:12,13])
    ax_map = plt.subplot(gs[0:12,1:13])
    ax_UWD = plt.subplot(gs[17:22,1:-1])
    ax_SDH = plt.subplot(gs[23:,1:-1])
    cf = ax_map.contourf(X,Y,uztot[i,:,:],levels = np.linspace(np.min(uztot),0,30))
    cb = Colorbar(ax = ax_cb, mappable = cf,ticks = np.linspace(np.min(uztot),0,5))
    cbar_label = np.round(np.linspace(np.min(uztot),0,5)*10)/10
    cb.ax.set_yticklabels(cbar_label,fontsize = 6 )
    cb.set_label('Vert. Displacement [m]',fontsize = 8 )

    ax_map.set_xticks([np.min(X),0,np.max(X)])
    ax_map.set_yticks([np.min(Y),0,np.max(Y)])
    ax_map.set_xlabel('x [km]',fontsize = 8 )
    ax_map.set_ylabel('y [km]', fontsize = 8)
    ax_map.set_xticklabels(labels = [int(np.min(X)),int(0),int(np.max(X))], fontsize = 6)
    ax_map.set_yticklabels(labels = [int(np.min(Y)),0,int(np.max(Y))], fontsize = 6)

    
    h1 = []
    h2 = []
    names_st = ['ST1','ST2']
    for j in np.arange(len(cordx)):
        x = cordx[j]
        y = cordy[j]
        east = tilt[j,i,0]
        north = tilt[j,i,1]
        
        east_new = east /(east**2 + north**2)**0.5
        north_new = north / (east**2 + north**2)**0.5
        east =east_new
        north = north_new
        handle1 = draw_vector([x,y],[x + east * npix,y + north*npix ],'orange', ax_map)
        handle1 = draw_vector([x,y],[x - east * npix,y - north*npix ],'orange', ax_map)
        ax_map.text(x-2,y + 1, names_st[j],fontsize = 6)
    ax_map.plot(xSource / 1000,ySource /1000,'kx',markersize = 1)
    ax_map.plot(cordx ,cordy ,'bo', markersize = 1)
    timeseries1 = ax_UWD.plot(time_tilt,magnitude1 *1e+6,'blue')
    timeseries2 = ax_SDH.plot(time_tilt,magnitude2*1e+6,'blue')
    time_segment = time_tilt[:i]
    magnitude_segment1 = magnitude1[:i]
    magnitude_segment2 = magnitude2[:i]
    
    ax_UWD.plot(time_segment,magnitude_segment1*1e+6,'r')
    ax_UWD.set_xlim([np.min(time_tilt),np.max(time_tilt)])
    tck = np.round ((np.linspace(np.min(magnitude1),0,3)*1e+6) / 10) * 10
    tck = tck.astype(int)
    ax_UWD.set_yticks(tck)
    ax_UWD.set_yticklabels(labels = tck,fontsize = 6)
    ax_UWD.set_xticklabels(labels = [])
    ax_UWD.set_ylabel(names_st[0] + ' [$\mu$rad]',fontsize = 8)

    ax_SDH.plot(time_segment,magnitude_segment2*1e+6,'r')
    ax_SDH.set_xlim([np.min(time_tilt),np.max(time_tilt)])
    tck =np.round((np.linspace(np.min(magnitude2),0,3)*1e+6) / 10) * 10
    tck = tck.astype(int)
    ax_SDH.set_yticks(tck)
    ax_SDH.set_yticklabels(labels = tck,fontsize = 6)
    ax_SDH.tick_params(axis='x', labelsize= 6 )
    ax_SDH.set_ylabel(names_st[1]+ ' [$\mu$rad]',fontsize = 8)
    ax_SDH.set_xlabel('Time [Days]',fontsize = 8)
    if len(str(filename_counter)) == 1:
        filename = '000' + str(filename_counter)
    elif len(str(filename_counter)) == 2:
        filename = '00' + str(filename_counter)
    elif len(str(filename_counter)) == 3:
        filename = '0' + str(filename_counter)
    elif len(str(filename_counter)) == 4:
        filename = str(filename_counter)
    plt.savefig('movie/' + str(filename) + '.png',dpi = 300)
    filename_counter = filename_counter + 1
    plt.close('all')
    
