# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:43:17 2022

@author: afisher
"""
import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import matplotlib.pyplot as plt


file = '2022-07-18 corrected_final_trajectory.csv'
data = pd.read_csv(file)

meas = data[['timex','x']].rename(columns={'timex':'time', 'x':'data'}).copy()
xamps = pwf.get_measurement_amplitudes(meas, annotate_plot=False, ref_magnet=False)

meas = data[['timey','y']].rename(columns={'timey':'time', 'y':'data'}).copy()
yamps = pwf.get_measurement_amplitudes(meas, annotate_plot=False, ref_magnet=False)


data.x = (data.x-data.x.iloc[0])/xamps.mean()
data.y = (data.y-data.y.iloc[0])/yamps.mean()

fig, ax = plt.subplots()
ax.plot(data.timex, data.x, label='X')
ax.plot(data.timey, data.y, label='Y')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Norm. Amplitude')
ax.set_xlim([0, .004])
ax.set_title('Final Corrected Trajectory')