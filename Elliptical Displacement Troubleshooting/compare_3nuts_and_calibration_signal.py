# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:44:50 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')


ref = pd.read_csv('calibration_signal.csv')
nuts = pd.read_csv('3nuts_signal.csv')

xdata = ref[['time','x']].rename(columns={'x':'data'})
ydata = ref[['time','y']].rename(columns={'y':'data'})
ref_xamp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000
ref_yamp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000

xdata = nuts[['time','x']].rename(columns={'x':'data'})
ydata = nuts[['time','y']].rename(columns={'y':'data'})
nuts_xamp = pwf.get_measurement_amplitudes(xdata, ref_magnet=False).mean()*1000
nuts_yamp = pwf.get_measurement_amplitudes(ydata, ref_magnet=False).mean()*1000

fig, ax = plt.subplots()
ax.plot(ref.time, ref.x-ref.x.iloc[0], label='ref')
ax.plot(nuts.time, nuts.x-nuts.x.iloc[0], label='nuts')
ax.legend()
ax.set_ylabel('X traj')

fig, ax = plt.subplots()
ax.plot(ref.time, ref.y-ref.y.iloc[0], label='ref')
ax.plot(nuts.time, nuts.y-nuts.y.iloc[0], label='nuts')
ax.legend()
ax.set_ylabel('Y traj')