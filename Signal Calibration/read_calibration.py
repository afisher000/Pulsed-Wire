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

# Voltage was set to 10 (black button, x1), but current was measured at 3.2 volts?

# Get calibration fits
cal_data = pd.read_csv('calibration_data.csv')
x_fit = np.polyfit(cal_data.dist, cal_data.xvolts, 1)
y_fit = np.polyfit(cal_data.dist, cal_data.yvolts, 1)
x_cal = x_fit[0]*1000 #mV/um
y_cal = y_fit[0]*1000 #mV/um

# Plot calibration fits
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
axes[0].scatter(cal_data.dist, cal_data.xvolts, label='Meas')
axes[0].plot(cal_data.dist, np.polyval(x_fit, cal_data.dist), label=f'Fit: {x_cal:.1f} mV/um')
axes[1].scatter(cal_data.dist, cal_data.yvolts, label='Meas')
axes[1].plot(cal_data.dist, np.polyval(y_fit, cal_data.dist), label=f'Fit: {y_cal:.1f} mV/um')
axes[1].set_xlabel('Displacement (um)')
axes[0].set_ylabel('XVolts')
axes[1].set_ylabel('YVolts')
axes[0].legend()
axes[1].legend()

# Get peak to peak amplitudes
meas = pd.read_csv('calibration_signal.csv')
xdata = meas[['time','x']].rename(columns={'x':'data'})
ydata = meas[['time','y']].rename(columns={'y':'data'})

x_amp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False, annotate_plot=True).mean()*1000 #mV
y_amp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False, annotate_plot=True).mean()*1000 #mV

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
axes[0].plot(xdata.time, xdata.data)
axes[1].plot(ydata.time, ydata.data)
axes[1].set_xlabel('Time (s)')
axes[0].set_ylabel('XVolts')
axes[1].set_ylabel('YVolts')

# Output results
print('''X Calibrations:
      x_calibration = %.1f mV/um
      x_amplitude = %.1f mV
      x_deflection = %.1f um''' % (x_cal, x_amp, x_amp/x_cal/2) )
      
print('''Y Calibrations:
      y_calibration = %.1f mV/um
      y_amplitude = %.1f mV
      y_deflection = %.1f um''' % (y_cal, y_amp, y_amp/y_cal/2) )
