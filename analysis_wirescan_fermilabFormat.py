# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:26:55 2022

@author: afisher
"""
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_pulsedwire import get_signal_means_and_amplitudes, low_pass_filter
import seaborn as sns
plt.close('all')

# folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 1\\2023-03-02 (xtraj, yoffset)'
# folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 1\\2023-03-02 (ytraj, xoffset)'
folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 2\\2023-03-03 (xtraj, yoffset) (3)'
# folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 2\\2023-03-03 (ytraj, xoffset) (2)'

traj_coord = folder[folder.find('traj')-1]
offset_coord = folder[folder.find('offset')-1]
dt = 1e-7

wirescan_df = pd.DataFrame()
# Loop over datasets in folder
for file in os.listdir(folder):
    # Skip calibration or non-csv files
    if (not file.startswith('(')) or (not file.endswith('.csv')):
        continue
    print(f'Reading {file}...')
    
    # Parse offset
    if offset_coord == 'x':
        offset = int(file[file.find('(')+1:file.find(',')])
    else:
        offset = int(file[file.find(',')+1:file.find(')')])
        
    # Compute mean and amplitudes from dataset
    signal = pd.read_csv(os.path.join(folder, file), header=None)[0]
    time = np.arange(0, len(signal)*dt, dt)

    means, amps = get_signal_means_and_amplitudes(time, signal, True)
    
    # Correct amplitude using calibration and compute statistics
    # amp_df['avg'] = amp_df.mean(axis=1)
    # amp_df['stdev'] = amp_df.std(axis=1)
    # amp_df['rel_stdev'] = amp_df.stdev/amp_df.avg
    print(f'Mean Voltage: {means.mean()}')
    # Add avg to wirescan_df
    wirescan_df[offset] = amps
    
wirescan_df = wirescan_df.transpose().sort_index()
    

def plot_wirescan_concavity(wirescan_df, peak_range=[0,5]):
    fig, ax = plt.subplots()
    min_peak = peak_range[0]
    max_peak = peak_range[1]
    wirescan_df.iloc[:,min_peak:max_peak].plot(ax=ax)
    ax.set_xlabel('Offset (um)')
    ax.set_ylabel('Amplitude (mV)')
    

def get_axis_fit(series):
    peak = series.name
    poly = np.polyfit(series.index, series.values, 2)
    axis = -poly[1]/(2*poly[0])
    residuals = series.values - np.polyval(poly, series.index)
    rmse = np.sqrt(np.sum(residuals**2))
    return peak, axis, rmse
    
def get_wire_adjustments(peaks, axis, weights, dz_p1_und = 41, dz_und_p2 = 12):
    dz_p1_und = 41 #inches
    dz_und_p2 = 12 #inches
    und_L = 39 #inches
    
    # Center on undulator
    zp1 = (-und_L/2 - dz_p1_und)*25400 #um
    zp2 = (und_L/2 + dz_und_p2)*25400 #um
    peaks_um = 32000/2*(peaks-peaks.mean())
    poly = np.polyfit(peaks_um, axis, 1, w=weights)
    
    # Compute and print adjustments
    p1_adjust = round(np.polyval(poly, zp1), -1) #nearest 10um
    p2_adjust = round(np.polyval(poly, zp2), -1) #nearest 10um
    print(f'P1 Adjustment: {p1_adjust}um')
    print(f'P2 Adjustment: {p2_adjust}um')
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('coolwarm')
    ax.scatter(peaks_um, axis, c=weights, cmap=cmap)
    ax.plot(peaks_um, np.polyval(poly, peaks_um))
    # ax.set_ylim([-1000,1000])
    return p1_adjust, p2_adjust
    
    
concavity_df = wirescan_df.apply(get_axis_fit).transpose()
concavity_df.columns = ['peak','axis','rmse']
cmap = plt.get_cmap('coolwarm')
concavity_df.plot.scatter('peak','axis', c='rmse')

# Get wire adjustments
peaks = concavity_df.peak.values
axis = concavity_df.axis.values
weights = 1/concavity_df.rmse.values
get_wire_adjustments(peaks, axis, weights)

plot_wirescan_concavity(wirescan_df, peak_range=[5,10])
plot_wirescan_concavity(wirescan_df, peak_range=[25,30])
plot_wirescan_concavity(wirescan_df, peak_range=[35,45])







# %%
