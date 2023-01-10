# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:34:23 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_pulsedwire_edited import get_signal_means_and_amplitudes, low_pass_filter
import os
from scipy.interpolate import interp1d
import scipy.constants as sc
from scipy.integrate import cumtrapz
plt.close('all')

# Inputs
traj = 'x'
mT = -20
m0 = 80
G0 = 430 # relativistic gamma
K0 = 2.23
ku = 2*np.pi/.032

# Read dataset
archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive'
wirescan_folder = '2022-12-22 xtraj, yoffset'
dataset = pd.read_csv(os.path.join(archive_folder, wirescan_folder, '(0,0).0.csv'))

# Get peaks
time = dataset.time.values
signal = dataset.drop(columns=['time']).mean(axis=1)
pk_times, pk_vals = get_signal_means_and_amplitudes(time, signal, return_peaks=True)

# Linear fit from time(ms) to magnet lattice(m)
map_time_2_m = np.polyfit(pk_times, 2*np.arange(len(pk_times)), 1)
time_m = np.polyval(map_time_2_m, time)
pk_m = np.polyval(map_time_2_m, pk_times)

# Add eigenfunction to field at specified m
field = np.zeros_like(time_m)
eigen = pd.read_csv('8mmNS.csv')
f = interp1d(m0+eigen.magnet.values, eigen.field.values, fill_value=0, bounds_error=False)
field = field + f(time_m)*mT/1000

# Numerically integrate for trajectory
pre_factor = sc.elementary_charge/G0/sc.electron_mass/sc.speed_of_light
dz = (time_m[1]-time_m[0])*.032/4
sign = 1 if traj=='x' else -1
traj_kick = sign*pre_factor*cumtrapz(cumtrapz(field, dx=dz), dx=dz)
traj_kick = np.pad(traj_kick, 1, 'symmetric')

# Scale trajectory to volts and add 
mean_amp = np.abs(np.diff(pk_vals)).mean()
r_max = K0/G0/ku
tuned_signal = signal + traj_kick/r_max*mean_amp

plt.plot(time, signal)
plt.plot(time, tuned_signal)



# plt.plot(time_m, signal)
# new_df = pd.DataFrame(np.vstack([time_m, signal]).T, columns=['time_m','signal'])
# new_df.to_csv('xtraj_with_kick.csv')




