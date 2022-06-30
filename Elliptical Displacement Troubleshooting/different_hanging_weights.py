# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:45:41 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')


files = ['0nuts_signal.csv',
         '2nuts_signal.csv',
         '3nuts_signal.csv']

nuts = [0,2,3]
x_def = []
y_def = []

xcal, ycal = pwf.get_linear_calibration('0in_calibration.csv')

for file in files:
    meas = pd.read_csv(file)
    xdata = meas[['time','x']].rename(columns={'x':'data'})
    ydata = meas[['time','y']].rename(columns={'y':'data'})
    
    
    x_amp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000
    y_amp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000

    x_def.append(x_amp/xcal/2)
    y_def.append(y_amp/ycal/2)

xfit = np.polyfit(nuts, x_def, 1)
yfit = np.polyfit(nuts, y_def, 1)

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].scatter(nuts, x_def)
axes[1].scatter(nuts, y_def)
axes[1].set_xlabel('Number of Nuts')
axes[0].set_ylabel('X deflection (um)')
axes[1].set_ylabel('Y deflection (um)')
axes[0].plot(nuts, np.polyval(xfit, nuts), label=f'Fit: {xfit[0]}')
axes[1].plot(nuts, np.polyval(yfit, nuts), label=f'Fit: {yfit[0]}')
axes[0].legend()
axes[1].legend()

fig, ax = plt.subplots()
ax.plot([0,10],[0,10],c='k', label='y=x')
ax.scatter(x_def, y_def, label='data')
ax.set_xlabel('X deflection (um)')
ax.set_ylabel('Y deflection (um)')