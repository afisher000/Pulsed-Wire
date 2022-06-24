# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""


import numpy as np
import pandas as pd
import scope_functions as sf
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
df = sf.read_measurements(scope_id, **params)
fig, ax = plt.subplots()
ax.plot(df.time, df.y-df.y[0])










    


