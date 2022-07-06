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


# Check signal amplitudes vs disp
disp = np.arange(-15,26,5)
yamps = np.zeros(len(disp))
ysigs = np.zeros(len(disp))
for j in range(len(disp)):
    file = f'{disp[j]}um signal.csv'
    meas = pd.read_csv(file)
    ydata = meas[['time','y']].rename(columns={'y':'data'})
    yamp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000 #mV
    ysigs[j] = ydata.data.iloc[0]
    yamps[j] = yamp
    print(f'{file}: yamp={yamp:.1f}')

fig, ax = plt.subplots()
ax.scatter(disp, yamps/max(yamps))
ax.set_ylabel('Signal Amplitude')
ax.set_xlabel('Displacement')

mask = yamps/max(yamps)>.9
poly = np.polyfit(disp[mask], ysigs[mask], 1)
fit = abs(poly[0]*1000)
fig, ax = plt.subplots()
ax.scatter(disp, ysigs)
ax.plot(disp[mask], np.polyval(poly, disp[mask]), label=f'Fit: {fit:.1f}')
ax.legend()

print(f'Deflection = {max(yamps)/2/fit} with 0.76A')