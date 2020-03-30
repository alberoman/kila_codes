#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:39:11 2019

@author: aroman
"""
from numpy import *
from numpy import max as npmax
def  llh2local(llh,origin):
    '''
    llh2local     xy=llh2local(llh,origin)
    
    %Converts from longitude and latitude to local coorindates
    %given an origin.  llh [lon; lat; height] and origin should
    %be in decimal degrees. Note that heights are ignored and
    %that xy is in km.
    %
    %     Copyright [C] 2015
    %     Email: eedpsb@leeds.ac.uk or davidbekaert.com
    %     With permission by Peter Cervelli, Jessica Murray 
    % 
    %     This program is free software; you can redistribute it and/or modify
    %     it under the terms of the GNU General Public License as published by
    %     the Free Software Foundation; either version 2 of the License, or
    %     [at your option] any later version.
    % 
    %     This program is distributed in the hope that it will be useful,
    %     but WITHOUT ANY WARRANTY; without even the implied warranty of
    %     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %     GNU General Public License for more details.
    % 
    %     You should have received a copy of the GNU General Public License along
    %     with this program; if not, write to the Free Software Foundation, Inc.,
    %     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
    %
    %-------------------------------------------------------------
    %   Record of revisions:
    %
    %   Date          Programmer            Description of Change
    %   ====          ==========            =====================
    %
    %   Sept 7, 2000  Peter Cervelli		Original Code
    %   Oct 20, 2000  Jessica Murray        Changed name from DM_llh2local to 
    %                                       llh2local for use with non-DM functions;
    %                                       Added to help message to clarify order
    %                                       of 'llh' [i.e., lon, lat, height].
    %   Dec. 6, 2000  Jessica Murray        Clarified help to show that llh 
    %                                       is a column vector
    %
    %
    %-------------------------------------------------------------
'''
#Set ellipsoid constants [WGS84]

    a=6378137.0
    e=0.08209443794970


    llh= float64(llh)*pi/180
    origin=float64(origin)*pi/180

#Do the projection

    z = not_equal(llh[1,:],0)
    not_z = equal(llh[1,:],0)
    dlambda=llh[0,z]-origin[0]

    M=a*((1-e**2/4-3*e**4/64-5*e**6/256)*llh[1,z] - 
         (3*e**2/8+3*e**4/32+45*e**6/1024)*sin(2*llh[1,z]) +
         (15*e**4/256 +45*e**6/1024)*sin(4*llh[1,z]) - 
         (35*e**6/3072)*sin(6*llh[1,z]))

    M0=a*((1-e**2/4-3*e**4/64-5*e**6/256)*origin[1] - 
          (3*e**2/8+3*e**4/32+45*e**6/1024)*sin(2*origin[1]) + 
          (15*e**4/256 +45*e**6/1024)*sin(4*origin[1]) - 
          (35*e**6/3072)*sin(6*origin[1]))
   
    N=a/sqrt(1-e**2*sin(llh[1,z])**2)
    E=dlambda*sin(llh[1,z])
    xy = zeros((2,len(z)))
    xy[0,z]=N*1./tan(llh[1,z])*sin(E)
    xy[1,z]=M-M0+N*1./tan(llh[1,z])*(1-cos(E))

#Handle special case of latitude = 0
    
    xy[0,not_z]=a*dlambda[~z];
    xy[1,not_z]=-M0;

#Convert to km
   
    xy=xy/1000;
    return xy

def local2llh(xy,origin):
    
    '''
        %local2llh     llh=local2llh(xy,origin)
        %
        %Converts from local coorindates to longitude and latitude 
        %given the [lon, lat] of an origin. 'origin' should be in 
        %decimal degrees. Note that heights are ignored and that 
        %xy is in km.  Output is [lon, lat, height] in decimal 
        %degrees. This is an iterative solution for the inverse of 
        %a polyconic projection.
        %
        %     Copyright (C) 2015
        %     Email: eedpsb@leeds.ac.uk or davidbekaert.com
        %     With permission by Peter Cervelli, Jessica Murray 
        % 
        %     This program is free software; you can redistribute it and/or modify
        %     it under the terms of the GNU General Public License as published by
        %     the Free Software Foundation; either version 2 of the License, or
        %     (at your option) any later version.
        % 
        %     This program is distributed in the hope that it will be useful,
        %     but WITHOUT ANY WARRANTY; without even the implied warranty of
        %     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        %     GNU General Public License for more details.
        % 
        %     You should have received a copy of the GNU General Public License along
        %     with this program; if not, write to the Free Software Foundation, Inc.,
        %     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
        %
        %-------------------------------------------------------------
        %   Record of revisions:
        %
        %   Date          Programmer            Description of Change
        %   ====          ==========            =====================
        %
        %   Aug 23, 2001  Jessica Murray        Clarification to help.
        %
        %   Apr 4, 2001   Peter Cervelli        Added failsafe to avoid
        %                                       infinite loop because of
        %                                       covergence failure.
        %   Sep 7, 2000   Peter Cervelli		Original Code
        %
        %-------------------------------------------------------------
    '''   
   # %Set ellipsoid constants (WGS84)
    a=6378137.0
    e=0.08209443794970

   #%Convert to radians / meters


    xy=float64(xy)*1000
    origin=float64(origin)*pi/180
#%Iterate to perform inverse projection

    M0=a*((1-e**2/4-3*e**4/64-5*e**6/256)*origin[1] - 
          (3*e**2/8+3*e**4/32+45*e**6/1024)*sin(2*origin[1]) + 
          (15*e**4/256 +45*e**6/1024)*sin(4*origin[1]) - 
          (35*e**6/3072)*sin(6*origin[1]))

    z=not_equal(xy[1,:],-M0)
    not_z=equal(xy[1,:],-M0)
    llh = zeros((2,len(z)))

    A=(M0+xy[1,z])/a
    B=xy[0,z]**2./a**2+A**2

    llh[1,z]=A

    delta=Inf

    c=0;
    while npmax(abs(delta))>1e-10:
        C=sqrt((1-e**2*sin(llh[1,z])**2))*tan(llh[1,z])
        
        M=a*((1-e**2/4-3*e**4/64-5*e**6/256)*llh[1,z] - 
             (3*e**2/8+3*e**4/32+45*e**6/1024)*sin(2*llh[1,z]) +
             (15*e**4/256 +45*e**6/1024)*sin(4*llh[1,z]) - 
             (35*e**6/3072)*sin(6*llh[1,z]))

        Mn=1-e**2/4-3*e**4/64-5*e**6/256 - \
         -2*(3*e**2/8+3*e**4/32+45*e**6/1024)*cos(2*llh[1,z]) + \
         4*(15*e**4/256 +45*e**6/1024)*cos(4*llh[1,z]) + \
         -6*(35*e**6/3072)*cos(6*llh[1,z])

        Ma=M/a
        delta=-(A*(C*Ma+1)-Ma-0.5*(Ma**2+B)*C)/ \
           (e**2*sin(2*llh[1,z])*(Ma**2+B-2*A*Ma)/(4*C)+(A-Ma)*(C*Mn-2/sin(2*llh[1,z]))-Mn)
        llh[1,z]=llh[1,z]+delta
        c=c+1;
        if c>100:
            print('Convergence failure.')
  
   

    llh[0,z]=(arcsin(xy[0,z]*C/a))/sin(llh[1,z])+origin[0]

#%Handle special case of latitude = 0

    llh[0,not_z]=xy[0,not_z]/a+origin[0]
    llh[1,not_z]=0

#%Convert back to decimal degrees

    llh=llh*180/pi
    return llh

