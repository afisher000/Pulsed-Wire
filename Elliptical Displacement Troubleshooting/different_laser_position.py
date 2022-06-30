# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:58:04 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')


files = ['17in closer', '0in', '6in farther']
laser_positions = [-17, 0, 6]
wire_deflections = []

for file in files:
    calibration_file = file+'_calibration.csv'
    signal_file = file+'_signal.csv'
    
    # Get calibration
    cal = pwf.get_linear_calibration(calibration_file, plot=True)[0]
    
    # Get amplitude
    meas = pd.read_csv(signal_file)
    xdata = meas.rename(columns={'x':'data'})
    xamp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False, annotate_plot=True).mean()*1000
    
    wire_deflections.append(xamp/cal/2)
    print('''Data point
      file = %s
      calibration = %.1f mV/um
      amplitude = %.1f mV
      deflection = %.1f um''' % (file, cal, xamp, xamp/cal/2) )
    
fig, ax = plt.subplots()
ax.scatter(laser_positions, wire_deflections)
ax.set_xlabel('Laser Position (in)')
ax.set_ylabel('Wire Deflection (um)')


