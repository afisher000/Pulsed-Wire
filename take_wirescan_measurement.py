# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import oscilloscope as osc
import pulsedwire as pwf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

folder = '2022-09-02 (ytraj, xoffset)'
file = 'undulator_preliminary_trajectory.csv'

# Take scope measurement
params = {
    'max_meas':20,
    'channel_map':{'ch1':'x', 'ch2':'y'},
    'filename':file #os.path.join(folder, file)
    }

scope_id = 'USB0::0x699::0x408::C031986::INSTR'
scope = osc.setup_scope(scope_id, npoints=100000)
df = osc.get_measurements(scope, **params)
df.plot(x='time', y='y')











    


