# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 10:23:22 2022

@author: afisher
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_pulsedwire_edited as up
from scipy.signal import find_peaks, savgol_filter
plt.close('all')

# # Select Wirescan Folder
# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-22 xtraj, yoffset'
wirescan_folder = '2022-12-28 xtraj, yoffset'

# Get time, signal for on-axis measuremtn
archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive'
folder = os.path.join(archive_folder, wirescan_folder)
file = os.path.join(folder, '(0,0).0.csv')
df = pd.read_csv(file)

time = df.time.values
signal = df.drop(columns=['time']).mean(axis=1).values

# Get Pulsedwire field and peaks
pw_time, pw_field = up.get_field_from_trajectory(time, signal, fix_dispersion=False)


clip = int(.001/(pw_time[1]-pw_time[0]))
pw_time = pw_time[clip:-clip]
pw_field = pw_field[clip:-clip]
plt.plot(pw_time, pw_field)
pw_idx, _ = find_peaks(np.abs(pw_field), prominence = 0.5*np.abs(pw_field).max())


# Get Hall probe field and peaks
hall_probe_folder = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Undulator\\Undulator Tuning Code (Cleaned)'
hp_data = pd.read_csv(os.path.join(hall_probe_folder, 'hall_probe_fields.csv'))
hp_m = hp_data.m.values
hp_field = hp_data.By.values
hp_idx, _ = find_peaks(np.abs(hp_field), prominence = 0.5*np.abs(hp_field).max())
plt.plot(hp_m, hp_field)

# Use linear fits to transform pulsed_wire to magnet lattice and tesla fields
pw_idx = pw_idx[:len(hp_idx)]


time_poly = np.polyfit(pw_time[pw_idx], hp_m[hp_idx], 1)
pw_m = np.polyval(time_poly, pw_time)

scale_poly = np.polyfit(pw_m[pw_idx], hp_field[hp_idx]/pw_field[pw_idx], 1)
pw_scaled_field = pw_field * np.polyval(scale_poly, pw_m)

# Plot
plt.plot(hp_m, hp_field, label='hall probe')
plt.plot(pw_m, pw_scaled_field, alpha=0.5, label='pulsed wire')
# Plot errors
errors = (pw_scaled_field[pw_idx]-hp_field[hp_idx])
plt.scatter(pw_m[pw_idx], errors*10, c='r', s=15, label='Errors*10')
plt.legend()


# 
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(pw_time, pw_field)
ax[1].plot(hp_m, hp_field)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Volts/Time^2')
ax[0].set_title('Pulsed Wire')
ax[1].set_xlabel('Magnet')
ax[1].set_ylabel('Field (T)')
ax[1].set_title('Hall probe')
    