# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:44:22 2022

@author: afisher
"""

# See if we can parse nonlinearity of measurement
''' Amplitude is given by the quadratic calibration fit times an offaxis 
scaling factor: amp = (A_0 + a_fit(zero-zero_0)**2) * (1+alpha/2*ku^2*(offset-axis)**2).
If we fit our data to an axis, zero_0, a_fit, and A_0, will it match well to 
what we measure?'''

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import scipy.optimize
import numpy as np
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')

# Read in data
folder = '2022-07-12 ytraj calibration'
cal_data = pwf.read_calibration(path=folder, plot=False)

file = '2022-07-12 (ytraj, xoffset).csv'
data, peak_data = pwf.analyze_wirescan(file, plot=False)

# Get trajectory and offset coordinates
traj = file[file.find('traj')-1]
offset = file[file.find('offset')-1]

# Apply quadratic field to calibration data
trun_cal_data = cal_data[cal_data.amps>cal_data.amps.max()*.7]
cal_fit = np.polyfit(trun_cal_data.zeros, trun_cal_data.amps/1000, 2)

# Get predicted amplitude, compute relative error
ku = 2*np.pi/32000
if traj==offset:
    alphabeta = 1.16
    sign = -1
else:
    alphabeta = 1.53
    sign = 1
    
data['axis'] = peak_data.axis.loc[data.index]
offaxis_scale = 1 + sign*(alphabeta*ku*(data.offset-data.axis))**2/2
data['pred_amps'] = np.polyval(cal_fit, data.means) * offaxis_scale
data['rel_error'] = (data.amps - data.pred_amps)/data.pred_amps

# Make plots
fig, ax = plt.subplots()
ax.scatter(data.means, data.amps, label='Measured')
ax.scatter(data.means, data.pred_amps, label='Predicted')
ax.legend()
ax.set_xlabel('Means')
ax.set_ylabel('Amplitudes')


# Relative error by peak
data.groupby(level=0).rel_error.mean().plot(figure=plt.figure(), 
                                              xlabel='Peak Number',
                                              ylabel='Mean Relative error')
data.groupby(level=0).rel_error.std().plot(figure=plt.figure(), 
                                              xlabel='Peak Number',
                                              ylabel='Std Relative error')

# Relative error by offset
data.groupby('offset').rel_error.mean().plot(figure=plt.figure(), 
                                              xlabel='Offset',
                                              ylabel='Mean Relative error')
data.groupby('offset').rel_error.std().plot(figure=plt.figure(), 
                                              xlabel='Offset',
                                              ylabel='Std Relative error')

