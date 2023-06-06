# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:23:23 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')


# Loop over offsets
IDX = 4000 # Force signal to be same at this index



def get_reference(folder):
    file = '(0,0).csv'
    df = pd.read_csv(os.path.join(folder, file))
    time = df.time.values
    ref_signal = df.drop(columns=['time']).mean(axis=1)
    ref_signal = ref_signal - ref_signal[IDX]
    return ref_signal
    
def get_signal(ax, folder, file, subtract_ref = False):
        df = pd.read_csv(os.path.join(folder, file))
        time = df.time.values
        signal = df.drop(columns=['time']).mean(axis=1)
        if subtract_ref:
            signal = signal - get_reference(folder)
            signal = signal - signal[IDX]
        else:
            signal = signal - signal[IDX]
        return time, signal
    
def plot_offaxis(ax, folder, files, subtract_ref = False):
    for file in files:
        time, signal = get_signal(ax, folder, file, subtract_ref=subtract_ref)
        ax.plot(time, signal, label=file.strip('.csv'))
    ax.legend()
    
def plot_traj_sum(ax, folder, dx_files, dy_files):
    sum_signal = np.zeros(10000)
    for dx_file in dx_files:
        time, signal = get_signal(ax, folder, dx_file, subtract_ref=False)
        sum_signal += signal.values
        # ax.plot(signal.values)
        
    for dy_file in dy_files:
        time, signal = get_signal(ax, folder, dy_file, subtract_ref=False)
        sum_signal += signal.values
        # ax.plot(signal.values)
    
        
    ax.plot(sum_signal)
    ax.set_ylabel('SUM of trajectories')
    return
        
    
    
    
# archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive\\THESEUS 1 PulsedWire Data'
archive_folder = ''

offaxis_folder = '2023-06-01 offaxis'
offaxis_folder = 'Alignments5'


offaxis_folder = '2023-06-01 ytraj, xoffset'
# offaxis_folder = '2023-06-01 xtraj, yoffset'
# offaxis_folder = '2023-06-05 ytraj, xoffset'
# offaxis_folder = '2023-06-05 xtraj, yoffset'


folder = os.path.join(archive_folder, offaxis_folder)


# offsets = [-500,500]
# dx_files = [f'({dx},0).csv' for dx in offsets]
# dy_files = [f'(0,{dy}).csv' for dy in offsets]

# xfolder = os.path.join(archive_folder, offaxis_folder, 'xtraj')
# yfolder = os.path.join(archive_folder, offaxis_folder, 'ytraj')


# for subtract_ref in [True, False]:
#     fig, ax = plt.subplots(nrows=2, ncols=2)
#     plot_offaxis(ax[0,0], xfolder, dx_files, subtract_ref=subtract_ref)
#     plot_offaxis(ax[0,1], xfolder, dy_files, subtract_ref=subtract_ref)
#     plot_offaxis(ax[1,0], yfolder, dx_files, subtract_ref=subtract_ref)
#     plot_offaxis(ax[1,1], yfolder, dy_files, subtract_ref=subtract_ref)
#     ax[0,0].set_ylabel('Xtraj')
#     ax[1,0].set_ylabel('Ytraj')
#     ax[1,0].set_xlabel('Xoffset')
#     ax[1,1].set_xlabel('Yoffset')

# fig, ax = plt.subplots(nrows=2)
# plot_traj_sum(ax[0], xfolder, dx_files, dy_files)
# plot_traj_sum(ax[1], yfolder, dx_files, dy_files)
# ax[0].set_ylabel('Summed X Traj')
# ax[1].set_ylabel('Summed Y Traj')


offsets = [-1500, -1000, -500, 0, 500, 1000, 1500]
dx_files = [f'({dx},0).csv' for dx in offsets]
dy_files = [f'(0,{dy}).csv' for dy in offsets]

folder = os.path.join(archive_folder, offaxis_folder)
files = dx_files

for subtract_ref in [True, False]:
    fig, ax = plt.subplots()
    plot_offaxis(ax, folder, files, subtract_ref=subtract_ref)
    ax.set_ylabel('Ytraj')
    






