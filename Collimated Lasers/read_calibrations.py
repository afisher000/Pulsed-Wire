# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:51:53 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')


# Check consistency of calibrations
files = ['calibration.csv', 'calibration2.csv']
print('\nCheck calibration consistency')
for file in files:
    x_cal, y_cal = pwf.get_linear_calibration(file, plot=True)
    print(f'{file}: xcal={x_cal:.1f}, ycal={y_cal:.1f}')

# Check signal amplitudes vs disp
print('\nCheck signal amplitudes:')
files = ['-20um_signal.csv','-10um_signal.csv','0um_signal.csv',
         '10um_signal.csv','20um_signal.csv']
for file in files:
    meas = pd.read_csv(file)
    xdata = meas[['time','x']].rename(columns={'x':'data'})
    x_amp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000 #mV
    print(f'{file}: xamp={x_amp:.1f}')

# Check signal amplitudes repeatability
print('\nCheck signal repeatability at 0um)')
files = ['signal1.csv','signal2.csv','signal3.csv']
for file in files:
    # Get calibration fits
    x_cal, y_cal = pwf.get_linear_calibration('calibration.csv')
    
    # Get peak to peak amplitudes
    meas = pd.read_csv(file)
    xdata = meas[['time','x']].rename(columns={'x':'data'})
    ydata = meas[['time','y']].rename(columns={'y':'data'})
    x_amp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000 #mV
    y_amp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000 #mV

    print(f'{file}: xamp={x_amp:.1f}, yamp={y_amp:.1f}')
