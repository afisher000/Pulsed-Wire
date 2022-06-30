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
x_cal, y_cal = pwf.get_linear_calibration('calibration_data.csv')

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
