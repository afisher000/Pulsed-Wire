# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import scope 
import pulsedwire as pwf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# folder = '2022-07-13 (ytraj, xoffset)'
file = 'dispersion_mid2(-5in).csv'

# Take scope measurement
params = {
    'max_meas':30,
    'channel_map':{'ch2':'y'},
    'rep_rate':1.0,
    'filename':file
    }
scope_id = 'USB0::0x699::0x408::C031986::INSTR'
df = scope.read_measurements(scope_id, **params)











    


