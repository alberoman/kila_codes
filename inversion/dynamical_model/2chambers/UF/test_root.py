#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:48:47 2020

@author: aroman
"""

from main_lib import *
params = [0.005223910705580266, 0.4437959540268115, 0, 1.2626042129151607e-05 ,0.006415497169061922, 1.0064279622412378, -0.5032140616055956 ,0.5032139811206189, 0.49679856443653364]
TSLIP = 0
TSLIP_seed = 1
tseg,tslip,PS,PD,pd0 = TwoChambers_UF(np.array([0.1,0.1]),params,0,TSLIP,TSLIP_seed) # Calculate ps at the first cycle