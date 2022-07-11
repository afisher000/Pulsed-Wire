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
disp = np.array([-20,-10,-5,0,5,10,20])
xamps = np.zeros(len(disp))
xsigs = np.zeros(len(disp))
for j in range(len(disp)):
    file = f'{disp[j]}um.csv'
    meas = pd.read_csv(file)
    xdata = meas[['time','x']].rename(columns={'x':'data'})
    xamp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000 #mV
    xsigs[j] = xdata.data.iloc[0]
    xamps[j] = xamp
    print(f'{file}: xamp={xamp:.1f}')

fig, ax = plt.subplots()
ax.scatter(disp, xamps/max(xamps))
ax.set_ylabel('Signal Amplitude')
ax.set_xlabel('Displacement')

mask = xamps/max(xamps)>.9
poly = np.polyfit(disp[mask], xsigs[mask], 1)
fit = abs(poly[0]*1000)
fig, ax = plt.subplots()
ax.scatter(disp, xsigs)
ax.plot(disp[mask], np.polyval(poly, disp[mask]), label=f'Fit: {fit:.1f}')
ax.legend()

print(f'Deflection = {max(xamps)/2/fit} with 1.95A')