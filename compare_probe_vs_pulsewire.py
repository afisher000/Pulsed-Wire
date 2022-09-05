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


def df_from_files(files):
    ''' Gather data from files and return as struct with columns including
    ['x', 'timex','y','timey']. If only one file, it is assumed the columns 
    include ['time','x','y'] already. If two files, it is assumed they are
    wirescan files for xtraj and ytraj, respectively. The data for the offset=0
    scan is returned.'''
    
    if not isinstance(files,list):
        files=[files]
        
    if len(files)==1:
        df = pd.read_csv(files[0])
        df['timex'] = df.time
        df['timey'] = df.time
        return df
    elif len(files)==2:
        xfile, yfile = files
        
        offset = xfile[xfile.find('offset')-1] + 'offset'
        xdata = pd.read_csv(xfile)
        xdata = xdata[xdata[offset]==0].drop(['iteration',offset], axis=1).reset_index(drop=True)
        
        offset = yfile[yfile.find('offset')-1] + 'offset'
        ydata = pd.read_csv(yfile)
        ydata = ydata[ydata[offset]==0].drop(['iteration',offset], axis=1).reset_index(drop=True)
        
        return xdata.join(ydata, lsuffix='x', rsuffix='y')
    else:
        print('Type of data returned from files not recognized.')
        return
    

xtraj_file = '2022-07-11 (xtraj, yoffset).csv'
ytraj_file = '2022-07-11 (ytraj, xoffset).csv'
# pw = df_from_files([xtraj_file, ytraj_file])
# pw = df_from_files('2022-05-10 final_trajectory.csv')
# pw = df_from_files('final_trajectory_shortened.csv')
# pw = pd.read_csv('2022-07-18 final_trajectory.csv')
pw = pd.read_csv('2022-07-18 corrected_final_trajectory.csv')

hp = df_from_files('hall_probe_trajectories.csv')


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
    est_pktimes, _ = pwf.get_measurement_pktimes_from_derivative(df['time'+coord], df[coord+'p'])
    period = np.diff(est_pktimes).mean()
    
    # Get exact peak data
    peak_data = [pwf.polyfit_peak(df['time'+coord], df[coord], est_pktime, window=period/5) 
                 for est_pktime in est_pktimes]
    
    return np.array(peak_data)

#### X Trajectory ####
xmodel = LinearRegression()
xmodel.fit(format_peakdata_for_regression(hp, 'x'),
       format_peakdata_for_regression(pw, 'x'))
x_results = xmodel.predict(hp[['time','x']].values)
x_time = x_results[:,0]
x_hp_transform = x_results[:,1]
x_hpfit = interp1d(x_time, x_hp_transform, fill_value=None, bounds_error=False) 
x_pwfit = interp1d(pw.timex, pw.x, fill_value=None, bounds_error=False)

#### Y Trajectory ####
ymodel = LinearRegression()
ymodel.fit(format_peakdata_for_regression(hp, 'y'),
           format_peakdata_for_regression(pw, 'y'))
y_results = ymodel.predict(hp[['time','y']].values)
y_time = y_results[:,0]
y_hp_transform = y_results[:,1]
y_hpfit = interp1d(y_time, y_hp_transform, fill_value=None, bounds_error=False)
y_pwfit = interp1d(pw.timey, pw.y, fill_value=None, bounds_error=False)



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

# Write xtraj fits
data = np.hstack([x_time, x_hpfit(x_time), x_pwfit(x_time)]).reshape(3, len(x_time)).T
pd.DataFrame(data, columns=['time','hallprobe','pulsedwire']).to_csv('xtraj_comparison.csv')




