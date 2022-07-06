# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:51:53 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
import os
plt.close('all')

folder = 'Scan1'
# Check consistency of calibrations
file = 'calibration.csv'
y_cal = pwf.get_linear_calibration(os.path.join(folder,file), plot=True)

data = pd.read_csv(os.path.join(folder, file))
fit = np.polyfit(data.dist, data.yvolts, 1)
cal = np.abs(fit[0]*1000)
1/0

# Check signal amplitudes vs disp
print('\nCheck signal amplitudes:')
displacements = [-30,-20, -10, 0, 10, 20]
yamps = []
for disp in displacements:
    file = f'{disp}um signal.csv'
    meas = pd.read_csv(os.path.join(folder,file))
    ydata = meas[['time','y']].rename(columns={'y':'data'})
    yamp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000 #mV
    yamps.append(yamp)
    print(f'{file}: xamp={yamp:.1f}')

fig, ax = plt.subplots()
ax.scatter(displacements, yamps/max(yamps))
ax.set_ylabel('Signal Amplitude')
ax.set_xlabel('Displacement')
