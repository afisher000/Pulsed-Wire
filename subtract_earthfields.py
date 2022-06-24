# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:45:04 2022

@author: afish
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\Pulse Wire Python Code\\CLEANED FILES\\PythonPackages')
import numpy as np
import pandas as pd
#import scope
import pulsedwire as pw
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
plt.close('all')

# Take scope measurement
scope_params = {
    'max_meas':8,
    'channel_map':{'ch1':'x', 'ch2':'y'},
    'rep_rate':1.4,
    #'filename':'FinalTrajectory.csv'
    }
scope_id = 'USB0::0x699::0x408::C031986::INSTR'
#df = scope.read_measurements(scope_id, **scope_params)
df = pd.read_csv('final_trajectory.csv')

# Read in earth signal
earth = pd.read_csv(os.path.join('22-06-01 No Prebuncher','signal.csv'))

# Previous measurements
A_ref = 796 #mV
C_ref = 123.7 #mV/um
V_ref = 10

C_earth_x = 99/2 #mV/um
C_earth_y = 56/5 #mV/um
V_earth = 10

# Find mean amplitude of measured signal
scn_x = (df.time>=.0025)&(df.time<=.0043) # for x traj
scn_y = (df.time>=.0015)&(df.time<=.0032) # for y traj

x_signal = df[scn_x].reset_index(drop=True)
y_signal = df[scn_y].reset_index(drop=True)
del x_signal['y'], y_signal['x']
x_signal.columns = ['volts','time']
y_signal.columns = ['volts','time']

x_signal.plot(x='time',y='volts', title='X traj')
y_signal.plot(x='time', y='volts', title='Y traj')
x_amp = x_signal.volts.max() - x_signal.volts.min()
y_amp = y_signal.volts.max() - y_signal.volts.min()
x_peaks = find_peaks(x_signal.volts, prominence=x_amp/8, distance=100)[0]
y_peaks = find_peaks(y_signal.volts, prominence=y_amp/8, distance=100)[0]
A_meas_x = np.mean( pw.get_amplitudes(x_signal, x_peaks) )/2*1000
A_meas_y = np.mean( pw.get_amplitudes(y_signal, y_peaks) )/2*1000


# Find correction from earth signal
fx = interp1d(earth.time, savgol_filter(earth.x, 151, 3))
fy = interp1d(earth.time, savgol_filter(earth.y, 151, 3))

V_meas = 10
x_earth = fx(df.time) - fx(df.time[0])
y_earth = fy(df.time) - fy(df.time[0])

dx = x_earth * (V_ref*C_ref)/(V_earth*C_earth_x) * (A_meas_x/A_ref)
dy = y_earth * (V_ref*C_ref)/(V_earth*C_earth_y) * (A_meas_y/A_ref)


df['x_correct'] = df.x - dx
df['y_correct'] = df.y - dy


ax = df.plot(x='time', y=['x','x_correct'])
ax.plot(df.time, np.ones_like(df.time)*df.x[0], c='k')
ax = df.plot(x='time', y=['y', 'y_correct'])
ax.plot(df.time, np.ones_like(df.time)*df.y[0], c='k')









    


