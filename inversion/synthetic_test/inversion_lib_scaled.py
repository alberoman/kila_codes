#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:26:34 2019

@author: aroman
t"""
import numpy as np
import theano.tensor as tt
def piston_scaled(R,t):
    R5 = R
    R2 = 1e-6
    R3 = 1.8
    R1 = 0.05
    pressure = np.ones(len(t)) #init pressure array
    N = int(np.ceil((R3 - 1) * (R1 + 1) / (2 * (1 - R5) * R1)))
    freq = np.sqrt(1 + R1)
    dtslip = 3.14 / freq #deltaT slip
    tslip  = 0. #first time  slip
    frac = (1 - R5)/(1 + R1)
    one_plus_r1 = 1 + R1
    one_min_r5 = (1 - R5)
    for i in range(N+1):
        if i < 2:
            tslip = 0
        else:
            tslip = tslip + dtslip - 1. /R2 * np.log((R3 * one_plus_r1 - 2*(i - 1)*R1*one_min_r5 - one_plus_r1) / (R3 * one_plus_r1 - 2 *one_min_r5*(-1+R1*(i - 2)) - one_plus_r1))
        tstick  = tslip + dtslip
        pressure[t >= tslip] = frac * (1 - 2*R1*(i - 1) - np.cos(freq * (t[t >= tslip] - tslip))) -1
        pstick  = 2 * frac * (1- R1*(i - 1)) -1
        pressure[t > tstick] = (pstick + R3) * np.exp(-R2 * (t[t > tstick]-tstick)) - R3
    return pressure

# define a theano Op for our direct model
class direct_model(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dscalar] # expects a vector of parameter values when called
    otypes = [tt.dvector] # outputs a vector 
    
    def __init__(self, direct_model,x):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.model = direct_model
        self.x = x

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        R_vect, = inputs  # this will contain my variables
        # call the log-likelihood function
        press = self.model(R_vect, self.x)

        outputs[0][0] = np.array(press) # output the log-likelihood