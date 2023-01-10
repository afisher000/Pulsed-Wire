# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:58:34 2022

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




def get_dslope(dataset, m0=80, plot_tf=False):
    # Get peaks
    time = dataset.time.values
    signal = dataset.drop(columns=['time']).mean(axis=1)
    pk_times, pk_vals = get_signal_means_and_amplitudes(time, signal, return_peaks=True)
    
    # Compute means
    pk_amp = np.convolve(pk_vals, [-.5, 1, -.5], 'valid').mean()
    print(pk_amp)
    pk_means = np.convolve(np.pad(pk_vals,1,'symmetric'), [.25, .5, .25], 'valid')
    
    # Get idx of input magnet
    idx = m0//2
    
    polyfinal = np.polyfit(pk_times[idx+2:idx+5], pk_means[idx+2:idx+5], 1)
    polyinit = np.polyfit(pk_times[idx-4:idx-1], pk_means[idx-4:idx-1], 1)

    dslope = polyfinal[0]-polyinit[0]
    
    if plot_tf:
        fig, ax = plt.subplots()
        ax.plot(time,signal)
        ax.scatter(pk_times, pk_vals)
        ax.plot(pk_times, pk_means, 'k--')
        ax.plot(pk_times[idx+2:idx+5], pk_means[idx+2:idx+5], 'r')
        ax.plot(pk_times[idx-4:idx-1], pk_means[idx-4:idx-1], 'r')
        ax.scatter(pk_times[idx], pk_means[idx], c='r')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Volts')
        
    return dslope


# Input magnet
m0 = 80

# Read dataset
archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive'
wirescan_folder = '2022-12-22 xtraj, yoffset'
folder = os.path.join(archive_folder, wirescan_folder)

# Get offset_coord
offset_coord = folder[folder.find('offset')-1]

# Loop over files
offsets = []
kicks = []
for file in os.listdir(folder):
    # Skip calibration or non-csv files
    if (not file.startswith('(')) or (not file.endswith('.csv')):
        continue
    print(f'Reading {file}...')
    dataset = pd.read_csv(os.path.join(folder, file))

    
    # Parse offset
    if offset_coord == 'x':
        offset = int(file[file.find('(')+1:file.find(',')])
    else:
        offset = int(file[file.find(',')+1:file.find(')')])
    dslope = get_dslope(dataset, m0=m0, plot_tf=(offset==0))
    
    # Save to lists
    offsets.append(offset)
    kicks.append(dslope)



# Sort by offset
idx_sort = np.argsort(offsets)
offsets = np.array(offsets)
offsets = offsets[idx_sort]
kicks = np.array(kicks)
kicks = kicks[idx_sort]

fig, ax = plt.subplots()
ax.plot(offsets, kicks)
ax.set_xlabel('Y Offset (um)');
ax.set_ylabel('Change in slope (V/s)')
ax.set_title(f'Kick of magnet T{m0}')

