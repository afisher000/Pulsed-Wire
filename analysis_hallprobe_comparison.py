# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:14:15 2023

@author: afisher
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_pulsedwire as up
import seaborn as sns
import scipy.constants as sc
from scipy.signal import find_peaks, savgol_filter
plt.close('all')

# Define data files
undulator_num = 2
wirescan_folder = '2023-02-03 ytraj, xoffset'
hallprobe_file = '2023-02-02-11.07.csv'
savgol_window = 301
est_wave_speed = 260


# Archive folders
pulsedwire_archive_folder = f'C:\\Users\\afisher\\Documents\\FASTGREENS DATA ARCHIVE\\THESEUS {undulator_num} PulsedWire Data'
hallprobe_archive_folder = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Undulator\\Undulator Tuning Code (Cleaned)\\On-axis Field CSVs'
pulsedwire_path = os.path.join(pulsedwire_archive_folder, wirescan_folder, '(0,0).1.csv')
hallprobe_path = os.path.join(hallprobe_archive_folder, hallprobe_file)

# Infer flags
FLAGS = {}
FLAGS['trajectory'] = 'x' if 'xtraj' in wirescan_folder else 'y'
FLAGS['field'] = 'by' if 'xtraj' in wirescan_folder else 'bx'

# Read data
pulsedwire = up.read_and_average_pulsedwire_measurement(pulsedwire_path)
hallprobe = up.read_hallprobe_field(hallprobe_path, FLAGS['field'])

# Integrate hallprobe field
dz = hallprobe.z.diff().mean()
G = 430
prefactor = (-1)**(FLAGS['field']=='bx') * sc.elementary_charge*sc.speed_of_light/G/sc.electron_mass
hallprobe['velocity'] = prefactor * hallprobe.field.cumsum()*dz/sc.speed_of_light
hallprobe['trajectory'] = hallprobe.velocity.cumsum()*dz/sc.speed_of_light


# Differentiate pulsed wire 
dt = pulsedwire.time.diff().mean()
pulsedwire['trajectory'] = savgol_filter(pulsedwire.trajectory, savgol_window, 3)
pulsedwire['velocity'] = up.get_numerical_derivative(
    pulsedwire.time, pulsedwire.trajectory, filtertype='savgol', savgol_window=savgol_window
)
pulsedwire['field'] = up.get_numerical_derivative(
    pulsedwire.time, pulsedwire.velocity, filtertype='savgol', savgol_window=savgol_window
)

# Decrease in signal amplitude
def get_amp_decay(signal):
    means, amps = up.get_signal_means_and_amplitudes(np.arange(len(signal)), signal)
    return amps/amps[:4].mean()

traj_amps = get_amp_decay(pulsedwire.trajectory)
vel_amps = get_amp_decay(pulsedwire.velocity)
field_amps = get_amp_decay(pulsedwire.field)

fig, ax = plt.subplots()
ax.plot(traj_amps, label='traj')
ax.plot(vel_amps, label='vel')
ax.plot(field_amps, label='field')
ax.legend()


# Dispersion effect on ideal hall probe trajectory
hallprobe['time'] = hallprobe.z/est_wave_speed
disp_time, disp_signal = up.correct_dispersion(hallprobe.time, hallprobe.velocity, EIwT = 4e-7, reduce_dt=1, reduce_fmax=5, reduce_df=2)
fig, ax = plt.subplots()
ax.plot(hallprobe.time, hallprobe.velocity, label='ideal')
ax.plot(disp_time, disp_signal, label='with dispersion')
ax.legend()

fig, ax = plt.subplots()
ax.plot(get_amp_decay(hallprobe.velocity), label='ideal hp velocity')
ax.plot(get_amp_decay(disp_signal), label='with dispersion')
ax.legend()


# Get time_to_z linear map
abs_pw_vel = np.abs(pulsedwire.velocity)
pw_idx, _ = find_peaks(abs_pw_vel, prominence = 0.5*abs_pw_vel.max())

abs_hp_vel = np.abs(hallprobe.velocity)
hp_idx, _ = find_peaks(abs_hp_vel, prominence = 0.5*abs_hp_vel.max())

time_to_z_poly = np.polyfit(pulsedwire.time[pw_idx], hallprobe.z[hp_idx], 1)
def time_to_z(time_values):
    return np.polyval(time_to_z_poly, time_values)

scaling = abs_hp_vel.max()/abs_pw_vel.max()

fig, ax = plt.subplots()
ax.plot(time_to_z(pulsedwire.time), scaling*pulsedwire.velocity, label='pw')
ax.plot(hallprobe.z, hallprobe.velocity, label='hp')
ax.legend()


fig, ax = plt.subplots()
ax.plot(get_amp_decay(pulsedwire.trajectory), label='pw')
ax.plot(get_amp_decay(hallprobe.trajectory), label='hp')
ax.legend()

fig, ax = plt.subplots()
ax.plot(get_amp_decay(pulsedwire.velocity), label='pw')
ax.plot(get_amp_decay(hallprobe.velocity), label='hp')
ax.legend()

fig, ax = plt.subplots()
ax.plot(get_amp_decay(pulsedwire.field), label='pw')
ax.plot(get_amp_decay(hallprobe.field), label='hp')
ax.legend()




