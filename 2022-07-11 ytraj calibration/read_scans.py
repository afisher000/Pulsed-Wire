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
amps = np.zeros(len(disp))
zeros = np.zeros(len(disp))
for j in range(len(disp)):
    file = f'{disp[j]}um.csv'
    meas = pd.read_csv(file)
    data = meas[['time','y']].rename(columns={'y':'data'})
    amp = pwf.get_measurement_amplitudes(data, ref_magnet=False).mean()*1000 #mV
    zeros[j] = data.data.iloc[0]
    amps[j] = amp
    print(f'{file}: amp={amp:.1f}')

fig, ax = plt.subplots()
ax.scatter(disp, amps/max(amps))
ax.set_ylabel('Signal Amplitude')
ax.set_xlabel('Displacement')

mask = amps/max(amps)>.9
poly = np.polyfit(disp[mask], zeros[mask], 1)
fit = abs(poly[0]*1000)
fig, ax = plt.subplots()
ax.scatter(disp, zeros)
ax.plot(disp[mask], np.polyval(poly, disp[mask]), label=f'Fit: {fit:.1f}')
ax.legend()

print(f'Deflection = {max(amps)/2/fit} with 1.95A')