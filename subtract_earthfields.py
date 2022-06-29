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
import pulsedwire as pwf
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
plt.close('all')


df = pd.read_csv('final_trajectory.csv')

# Read in earth signal
path = ('C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\'
        'April 2022 Pulse Wire\\Pulsewire Measurement Archive')
earth = pd.read_csv(os.path.join(path, 
                                 '2022-06-01 No Prebuncher',
                                 'signal.csv'))

# Measurement calibration and voltage
A_ref = 796 #mV
C_ref = 123.7 #mV/um
V_ref = 10

# Earth calibrations and voltage
C_earth_x = 99/2 #mV/um
C_earth_y = 56/10 #mV/um
V_earth = 10

# Find mean amplitudes of measured signal
xdf = df[['time','x']].rename(columns={'x':'data'})
ydf = df[['time','y']].rename(columns={'y':'data'})
A_meas_x = pwf.get_measurement_amplitudes(xdf, ref_magnet=False).mean()/2*1000
A_meas_y = pwf.get_measurement_amplitudes(ydf, ref_magnet=False).mean()/2*1000

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









    


