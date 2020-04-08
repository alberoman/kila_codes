#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:45:43 2020

@author: aroman
"""
import numpy as np
from scipy import optimize

def system_2chambers_RK4(w0,t,deltat_RGK,par,pCrit,tol):
    #RK solvers with Newton-Raphson criteria for stopping integration 
    #defined by pCrit
    ps0,pd0 = w0
    t1,phi,r3 = par
    A = np.array([[- (1 + t1), + 1],[phi, - phi]])
    B = np.array([[-t1 * r3],[0]])
    ps = [ps0]
    pd = [pd0]
    time = [0] 
    vect = [[ps0],[pd0]]
    err = np.abs(ps0 - pCrit) 
    tstep = 0
    fx = ps0 - pCrit
    fxprime = np.dot(A[0,:],vect ) + B[0][0] 
    deltat_newt = - fx / fxprime
    hh = np.sign(deltat_newt) * min(deltat_newt,deltat_RGK)
    err = np.abs (ps[-1] - pCrit) / np.abs(pCrit)
    while err > tol:
        K1 = np.dot(A , vect) + B
        K2 = np.dot(A , vect + 0.5 * K1 * hh) + B
        K3 = np.dot(A , vect + 0.5 * K2 * hh) + B
        K4 = np.dot(A ,  vect +  K3 * hh) + B
        vect =  vect + 1. / 6  * (K1 + 2 * K2 + 2*K3 + K4) * hh
        ps.append(vect[0][0])
        pd.append(vect[1][0])
        time.append(time[-1] + hh[0])
        tstep = tstep + hh
        fx = ps[-1] - pCrit
        fxprime = np.dot(A[0,:],vect ) + B[0][0]  
        deltat_newt = - fx / fxprime
        hh= np.sign(deltat_newt) * min(deltat_newt,deltat_RGK)
        err = np.abs (ps[-1] - pCrit) / np.abs(pCrit)
    ps = np.array(ps)
    pd = np.array(pd)
    time = np.array(time)
    ps = ps[np.argsort(time)]
    pd = pd[np.argsort(time)]
    time = np.sort(time)
    return time,ps,pd

def ps_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    ps = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return  ps
def ps_analytic_smallVar(t,R3,T1,phi,pd0,ps0):
    ps = T1 * t * (-R3 - ps0) - (pd0 - ps0) * np.exp(-phi*t) / phi + (pd0 + phi * ps0 - ps0) / phi
    return ps
def pd_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def pd_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    pd = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return pd
def pd_analytic_smallVar(t,R3,T1,phi,pd0,ps0):
    pd = ps0 + (pd0 - ps0)*np.exp(-phi*t)
    return pd

def Twochambers_timein(w0,par,pslip,tslip,time,ps,pd,tsl_seed):
    ps0,pd0 = w0
    r3,t1,phi,a,b,c,d = par
    tsegment = time[time >= tslip]
    pssegment = ps_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    ps[time >= tslip] = pssegment
    pd[time >= tslip] = pdsegment
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(pd_analytic_root,0,1e+7, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    pdend = pd_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tslip,ps,pd,pdend

def system_2chambers(w0,par,pslip,tslip,tsl_seed):
    ps0,pd0 = w0
    r3,t1,phi,a,b,c,d = par
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tslip = optimize.brentq(pd_analytic_root,0,1e+7, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    tsegment = np.linspace(0,tslip,20)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    psend = pssegment[-1]
    return tsegment,tslip,pssegment,pdsegment,psend

def system_2chambers_smallVar(w0,par,nn):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi = par
    tslip = np.log((-2*r1*r5*nn + 2*r1*nn + r1*ps0 + r1 + 2*r5 + ps0 - 1)/(-2*r1*r5*nn + 2*r1*nn + r1*ps0 + r1 + ps0 + 1))/phi
    tsegment = np.linspace(0,tslip,30)
    pssegment = ps_analytic_smallVar(tsegment,r3,t1,phi,pd0,ps0)
    pdsegment = pd_analytic_smallVar(tsegment,r3,t1,phi,pd0,ps0)
    psend = pssegment[-1]
    return tsegment,tslip,pssegment,pdsegment,psend
    
                             



    
    