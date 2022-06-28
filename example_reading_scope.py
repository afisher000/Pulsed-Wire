# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import scope 
import pulsedwire as pwf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data_folder = '2022-06-02 (xtraj, yoffset) (2)'
file = '(0,-1000).2.csv'

# Take scope measurement
params = {
    'max_meas':30,
    'channel_map':{'ch2':'x'},
    'rep_rate':1.4,
    'filename':os.path.join(data_folder, file)
    }
scope_id = 'USB0::0x699::0x408::C031986::INSTR'
df = scope.read_measurements(scope_id, **params)











    


