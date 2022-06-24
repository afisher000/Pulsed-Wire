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


# Post positions (moved June 1)
#P1x = 5.80, P1y = 12.75
#P2x = 5.50, P2y = 10.25

# Adjustments:
# dP1x = 1.700, dP1y = -1.000
# dP2x = 0.070, dP2y = 0.700


# Post positions (moved June 2)
#P1x = 7.50, P1y = 11.75
#P2x = 5.57, P2y = 10.95

### INPUTS ###
#data_folder = '2022-06-01 (ytraj, xoffset)'
#data_folder = '2022-06-01 (xtraj, yoffset)'
#data_folder = '2022-06-02 (ytraj, xoffset)'
#data_folder = '2022-06-02 (xtraj, yoffset)'
#data_folder = '2022-06-02 (ytraj, xoffset) (2)'
data_folder = '2022-06-02 (xtraj, yoffset) (2)'


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
    'pks_end':32e-4,
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


# Map for data starting May 2 (reading scope data to laptop directly)
channel_map = {'time':'time', 'x':'x', 'y':'y'}

xbins = pd.Series({
    'pre':-np.inf,
    'ref1':9e-4,
    'ref1_end':13e-4,
    'ref2':21e-4,
    'ref2_end':25e-4,
    'pks':27e-4,
    'pks_end':43e-4,
    'post':np.inf
    })

ybins = pd.Series({
    'pre':-np.inf,
    'ref1':-2e-4,
    'ref1_end':3e-4,
    'ref2':11e-4,
    'ref2_end':13e-4,
    'pks':16e-4,
    'pks_end':32.5e-4,
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
    if file=='(1500,0).1.csv':
        ax_ref.plot(df.time, df.volts-df.volts[0])
        pwf.annotated_plot(df, bins, title=file)
    
    series = pwf.get_amplitudes_as_series(df, bins, noise_fac=2)
    
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


# Get Post adjustments
pk_offsets = qfit_axes[:12]
pk_idx = np.arange(len(pk_offsets))
p1_to_und = 1
und_to_p2 = .14
pwf.get_post_adjustments(pk_idx, pk_offsets, p1_to_und, und_to_p2)


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




