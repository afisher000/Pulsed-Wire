# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:24:52 2022

@author: afisher
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import pulsewire_functions as pwf
import datetime

plt.close('all')

### INPUTS ###
data_folder = '2022-06-01 (ytraj, xoffset) Andrew'

#data_folder = '2022-05-12 (ytraj, yoffset) Andrew'
#data_folder = '2022-05-10 (xtraj, yoffset) Andrew'
#data_folder = '2022-05-12 (ytraj, xoffset) Andrew'
#data_folder = '2022-05-10 (xtraj, xoffset) Jason 2'
GBL = pwf.parse_global_constants(data_folder)

# For data before May
channel_map = {'TIME':'time', 'CH1':'x', 'CH4':'y'}

xbins = pd.Series({
    'pre':-np.inf,
    'ref1':0e-4,
    'ref1_end':3e-4,
    'ref2':11e-4,
    'ref2_end':15e-4,
    'pks':17e-4,
    'pks_end':33e-4,
    'post':np.inf
    })

ybins = pd.Series({
    'pre':-np.inf,
    'ref1':2e-4,
    'ref1_end':7e-4,
    'ref2':15e-4,
    'ref2_end':20e-4,
    'pks':23.5e-4,
    'pks_end':37.8e-4,
    'post':np.inf
    })


# Map for data starting May 2 (reading data to laptop directly)
channel_map = {'time':'time', 'x':'x', 'y':'y'}



ybins = pd.Series({
    'pre':-np.inf,
    'ref1':-2e-4,
    'ref1_end':3e-4,
    'ref2':11e-4,
    'ref2_end':13e-4,
    'pks':15e-4,
    'pks_end':32e-4,
    'post':np.inf
    })



# Rest of code
GBL_map = {'time':'time', GBL.TRAJ:'volts'}

full_map = {key:GBL_map[value] 
            for key,value in channel_map.items()
            if value in GBL_map.keys()}

### CODE ###
fig_all, ax_all = plt.subplots()
fig_ref, ax_ref = plt.subplots()
bins = ybins if GBL.TRAJ=='y' else xbins

# Build dataframe for all files
pieces = []
for file in os.listdir(data_folder):
    #df = pd.read_csv(os.path.join(data_folder, file), skiprows=range(20))
    df = pd.read_csv(os.path.join(data_folder, file))
    df = df.rename(columns=full_map)[['time','volts']]
    
    
    ax_all.plot(df.time,df.volts-df.volts[0])
    if file=='(0,0).1.csv':
        ax_ref.plot(df.time, df.volts-df.volts[0])
    
    series = pwf.get_amplitudes_as_series(df, bins, noise_fac=2)
    #pwf.annotated_plot(df, bins, title=file)
    if series is not None:
        series['offset'] = pwf.get_offset_from_filename(data_folder, file)
        pieces.append(series)
    else: 
        print(f'Incorrect number of peaks for file {file}')
ax_all.set_title(f'{GBL.TRAJ}traj vs {GBL.OFF}')
fig_all.savefig(f'{GBL.TRAJ}traj vs {GBL.OFF} - allplots.jpg')
ax_ref.set_title(f'{GBL.TRAJ}traj on axis.jpg')
fig_ref.savefig(f'{GBL.TRAJ}traj on axis.jpg')

# Combine into one DataFrame
df = pd.DataFrame(pieces)
df.set_index('offset', inplace=True)
df = df.add_prefix('Peak')

# Fit concavities of each peak
qfit_axes, qfit_offsets = pwf.fit_concavity(df)
df = df/qfit_offsets #Normalize to offset

# Plot axes along undulator
pwf.plot_axis_along_undulator(GBL, qfit_axes, save=False)
pwf.plot_concavity(df, GBL, qfit_axes, pk_idxs=range(0,6))
pwf.plot_concavity(df, GBL, qfit_axes, pk_idxs=range(-6,0))



# Read dataframes, munge so columns are ['time','volts']
'''
hp = pd.read_csv('hall_probe_trajectories.csv')[['time',GBL.TRAJ]]
hp.rename(columns=GBL_map, inplace=True)
pw = pd.read_csv(os.path.join(data_folder,'(0,0).1.csv'), 
                    skiprows=range(20))
pw = pw.rename(columns=full_map)[['time','volts']]
pwf.compare_trajectories(hp, pw, bins, noise_fac=3, 
                         save=False, file=f'HP-PW Comparison - {GBL.TRAJ}traj.jpg')
'''




