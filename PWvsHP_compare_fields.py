# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 10:23:22 2022

@author: afisher
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_pulsedwire as up
from scipy.signal import find_peaks, savgol_filter
plt.close('all')

# # Select Wirescan Folder
# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-22 xtraj, yoffset'
# wirescan_folder = '2022-12-28 xtraj, yoffset'

# wirescan_folder = '2023-02-03 xtraj, yoffset'
# wirescan_folder = '2023-02-03 ytraj, xoffset'
# wirescan_folder = '2023-02-03 ytraj, xoffset (2)' #Clipped early on purpose to avoid mysterious kick at m=80
# wirescan_folder = '2023-02-06 xtraj, yoffset'
wirescan_folder = '2023-02-06 ytraj, xoffset'

# Get time, signal for on-axis measurement
archive_folder = 'C:\\Users\\afisher\\Documents\\FASTGREENS DATA ARCHIVE\\THESEUS 2 PulsedWire Data'
folder = os.path.join(archive_folder, wirescan_folder)
file = os.path.join(folder, '(0,0).1.csv')
df = pd.read_csv(file)

time = df.time.values
pw_traj = df.drop(columns=['time']).mean(axis=1).values
# time, pw_traj = up.correct_dispersion(time, pw_traj, c0=250, EIwT = 3.4e-9, reduce_dt=5, reduce_fmax=5, reduce_df=2)


# Get Pulsedwire field and peaks
c0 = 250
dt = time[1]-time[0]
dz = .4*.032 #smooth over 1/3 period with quintic
savgol_window = int(dz/(c0*dt)) if int(dz/(c0*dt))%2==1 else int(dz/(c0*dt))+1
pw_velocity = up.get_numerical_derivative(time, pw_traj, filtertype='savgol', savgol_window=savgol_window)
pw_field = up.get_numerical_derivative(time, pw_velocity, filtertype='savgol', savgol_window=savgol_window)
pw_time = time;


clip = int(.001/(pw_time[1]-pw_time[0]))
clip=1
pw_time = pw_time[clip:-clip]
pw_field = pw_field[clip:-clip]
pw_velocity = pw_velocity[clip:-clip]
pw_traj = pw_traj[clip:-clip]



# Get Hall probe field and peaks
hall_probe_folder = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Undulator\\Undulator Tuning Code (Cleaned)'
hp_data = pd.read_csv(os.path.join(hall_probe_folder, 'hall_probe_fields.csv'))
hp_m = hp_data.m.values
if 'xtraj' in wirescan_folder:
    hp_field = hp_data.By.values
    hp_velocity = np.cumsum(hp_field)
else:
    hp_field = -1*hp_data.Bx.values
    hp_velocity = np.cumsum(hp_field)
    hp_traj = np.cumsum(hp_velocity)


def align_hp_pw_signals(hp_signal, pw_signal, title):
    hp_signal = hp_signal - hp_signal.mean() #no entrance/exit field fringes
    pw_signal = pw_signal - pw_signal.mean()
    
    pw_max_idx, _ = find_peaks(pw_signal, prominence = 1.2*pw_signal.max())
    hp_max_idx, _ = find_peaks(hp_signal, prominence = 1.2*hp_signal.max())
    
    fig, ax = plt.subplots()
    ax.plot(pw_time, pw_signal)
    ax.scatter(pw_time[pw_max_idx], pw_signal[pw_max_idx])
    
    fig, ax = plt.subplots()
    ax.plot(hp_m, hp_signal)
    ax.scatter(hp_m[hp_max_idx], hp_signal[hp_max_idx])

    time_poly = np.polyfit(pw_time[pw_max_idx], hp_m[hp_max_idx], 1)
    pw_m = np.polyval(time_poly, pw_time)
    
    
    scales = hp_signal[hp_max_idx]/pw_signal[pw_max_idx]
    scale_poly = np.polyfit(pw_m[pw_max_idx], scales, 1)
    pw_scaled_signal = pw_signal * np.polyval(scale_poly, pw_m)
    hp_signal = hp_signal - hp_signal.mean()

    # Plot
    fig, ax = plt.subplots()
    ax.plot(hp_m, hp_signal, label='hall probe')
    ax.plot(pw_m, pw_scaled_signal, alpha=0.5, label='pulsed wire')
    
    # Plot errors
    errors = (pw_scaled_signal[pw_max_idx]-hp_signal[hp_max_idx])
    ax.scatter(pw_m[pw_max_idx], errors*10, c='r', s=15, label='Errors*10')
    ax.legend()
    ax.set_title(title)
    
# align_hp_pw_signals(hp_field, pw_field, 'Fields')
align_hp_pw_signals(hp_velocity, pw_velocity, 'Velocities')
# align_hp_pw_signals(hp_traj, pw_traj, 'Trajectory')
# 

# 
# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
# ax[0].plot(pw_time, pw_field)
# ax[1].plot(hp_m, hp_field)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Volts/Time^2')
# ax[0].set_title('Pulsed Wire')
# ax[1].set_xlabel('Magnet')
# ax[1].set_ylabel('Field (T)')
# ax[1].set_title('Hall probe')
    