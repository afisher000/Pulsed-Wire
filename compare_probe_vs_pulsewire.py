# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:23:20 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
plt.close('all')


pw = pd.read_csv('final_trajectory_shortened.csv')
#pw = pd.read_csv('May10 pulsedwire trajectories.csv')
pw.x = pw.x - pw.x.iloc[0]
pw.y = pw.y - pw.y.iloc[0]
hp = pd.read_csv('hall_probe_trajectories.csv')

def format_peakdata_for_regression(df, coord):
    ''' Does computations to find time, value pairs for peaks. The formatted 
    data can be used easily with a linear regression model to compare the 
    hall probe and pulsedwire trajectories.'''
    # Smooth
    window = len(df)/100 if len(df)/100%2 else len(df)/100+1
    df[coord] = savgol_filter(df[coord], round(window), 3)
    
    # Differentiate
    df[coord+'p'] = np.append(0, np.diff(df[coord]))
    
    # Get estimate peak times
    est_pktimes, _ = pwf.get_measurement_pktimes_from_derivative(df.time, df[coord+'p'])
    period = np.diff(est_pktimes).mean()
    
    # Get exact peak data
    peak_data = [pwf.polyfit_peak(df.time, df[coord], est_pktime, window=period/5) 
                 for est_pktime in est_pktimes[:19]]
    
    return np.array(peak_data)

#### X Trajectory ####
xmodel = LinearRegression()
xmodel.fit(format_peakdata_for_regression(hp, 'x'),
       format_peakdata_for_regression(pw, 'x'))
x_results = xmodel.predict(hp[['time','x']].values)
x_time = x_results[:,0]
x_hp_transform = x_results[:,1]
x_hpfit = interp1d(x_time, x_hp_transform, fill_value=None, bounds_error=False) 
x_pwfit = interp1d(pw.time, pw.x, fill_value=None, bounds_error=False)

#### Y Trajectory ####
ymodel = LinearRegression()
ymodel.fit(format_peakdata_for_regression(hp, 'y'),
           format_peakdata_for_regression(pw, 'y'))
y_results = ymodel.predict(hp[['time','y']].values)
y_time = y_results[:,0]
y_hp_transform = y_results[:,1]
y_hpfit = interp1d(y_time, y_hp_transform, fill_value=None, bounds_error=False)
y_pwfit = interp1d(pw.time, pw.y, fill_value=None, bounds_error=False)



# Plot results
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].plot(x_time, x_hpfit(x_time), label='Hallprobe')
axes[0,0].plot(x_time, x_pwfit(x_time), label='Pulsedwire')
axes[0,1].plot(y_time, y_hpfit(y_time), label='Hallprobe')
axes[0,1].plot(y_time, y_pwfit(y_time), label='Pulsedwire')
axes[1,0].plot(x_time, x_pwfit(x_time)-x_hpfit(x_time), label='PW-HP Error')
axes[1,1].plot(y_time, y_pwfit(y_time)-y_hpfit(y_time), label='PW-HP Error')

[ax.legend() for ax in axes.flatten()]
axes[1,0].set_xlabel('Scope Time (s)')
axes[1,1].set_xlabel('Scope Time (s)')
axes[0,0].set_ylabel('Signal (V)')
axes[1,0].set_ylabel('Signal (V)')





