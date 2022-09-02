# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:29:07 2022

@author: afish
"""

import pyvisa #need to install pyvisa with "pip install pyvisa"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_wfm_settings(scope):
    ''' Return a pandas Series object containing waveform (wfm) settings of scope.'''
    values = np.array([
        scope.query_ascii_values('wfmoutpre:xinc?')[0],
        scope.query_ascii_values('wfmoutpre:xzero?')[0],
        scope.query_ascii_values('wfmoutpre:ymult?')[0],
        scope.query_ascii_values('wfmoutpre:yoff?')[0],
        scope.query_ascii_values('wfmoutpre:yzero?')[0],
        scope.query_ascii_values('wfmoutpre:nr_pt?')[0]
        ])

    names = ['xinc','xzero','ymult','yoff','yzero','points']
    wfm = pd.Series(values, index=names)
    return wfm

# Parameters
npoints = 100
channel = 'ch1' #'ch1','ch2','ch3', or 'ch4'


# Set up the scope
scope_id = 'USB0::0x699::0x408::C031986::INSTR'
rm = pyvisa.ResourceManager()
scope = rm.open_resource(scope_id)

# Set timeout and what points to read
scope.timeout= 5000 
scope.write('data:start 1')
scope.write(f'data:stop {npoints}')

# Ask for the data as an array (integers from -128 to 128)
scope.write(f'data:source {channel}')
uint8_data = scope.query_binary_values('curve?', 
                datatype='b', is_big_endian=True, container=np.array)


# Scale uint8 data with waveform settings
wfm = get_wfm_settings(scope)
data = wfm.yzero + wfm.ymult*uint8_data
time = wfm.xzero + wfm.xinc*np.arange(wfm.points)

# Plot the data
fig, ax = plt.subplots()
ax.plot(time,data)
print(f'Read {len(data)} datapoints.')
