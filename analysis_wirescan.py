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
# archive_folder = 'C:\\Users\\afisher\\Documents\\FASTGREENS DATA ARCHIVE\\THESEUS 1 PulsedWire Data'
#archive_folder = 'C:\\Users\\afisher\\Documents\\FASTGREENS DATA ARCHIVE\\THESEUS 2 PulsedWire Data'
archive_folder = ''

# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-22 xtraj, yoffset'
# wirescan_folder = '2022-12-28 xtraj, yoffset'
# wirescan_folder = '2022-12-28 ytraj, xoffset'
# wirescan_folder = '2023-01-20 xtraj, yoffset'
# wirescan_folder = '2023-01-23 ytraj, xoffset'
# wirescan_folder = '2023-01-24 xtraj, yoffset'
# wirescan_folder = '2023-01-24 ytraj, xoffset'


# wirescan_folder = '2023-02-03 ytraj, xoffset'
# wirescan_folder = '2023-02-03 ytraj, xoffset'
# wirescan_folder = '2023-02-03 ytraj, xoffset (2)'
# wirescan_folder = '2023-02-06 xtraj, yoffset'
#wirescan_folder = '2023-02-06 ytraj, xoffset'

# THZ EXPERIMENT
# wirescan_folder = '2023-05-31 xtraj, yoffset'
wirescan_folder = '2023-06-01 ytraj, xoffset'

wirescan_folder = '2023-06-05 ytraj, xoffset'
wirescan_folder = '2023-06-05 xtraj, yoffset'
# 
folder = os.path.join(archive_folder, wirescan_folder)
#folder = wirescan_folder


folder = os.path.join(archive_folder, wirescan_folder)
traj_coord = folder[folder.find('traj')-1]
offset_coord = folder[folder.find('offset')-1]

# Get quartic fit of calibration curve
calibration = pd.read_csv(os.path.join(folder, 'calibration.csv'))
calibration = calibration.sort_values(by='voltage')
cal_poly = np.polyfit(calibration.voltage, calibration.amplitude, 4)

fig, ax = plt.subplots()
ax.scatter(calibration.voltage, calibration.amplitude, label='data')
ax.plot(calibration.voltage, np.polyval(cal_poly, calibration.voltage), label='Quartic Fit')
ax.set_ylabel('Amplitude (mV)')
ax.set_xlabel('Voltage (mV)')
ax.legend()

wirescan_df = pd.DataFrame()
# Loop over datasets in folder
for file in os.listdir(folder):
    # Skip calibration or non-csv files
    if (not file.startswith('(')) or (not file.endswith('.csv')):
        continue
    print(f'Reading {file}...')
    
    # Parse offset using regex
    if offset_coord == 'x':
        offset = int(file[file.find('(')+1:file.find(',')])
    else:
        offset = int(file[file.find(',')+1:file.find(')')])
        
    # Compute mean and amplitudes from dataset
    dataset = pd.read_csv(os.path.join(folder, file))
    mean_df = pd.DataFrame(columns = dataset.columns).drop(columns=['time'])
    amp_df = pd.DataFrame(columns = dataset.columns).drop(columns=['time'])
    for j in range(len(dataset.columns)-1):
        col = f'col{j}'
        time = dataset.time.values
        signal = dataset[col].values
        
        signal = low_pass_filter(time, signal, 4e4, 1)
        means, amps = get_signal_means_and_amplitudes(time, signal, plot_signal_peaks=False, plot_derivative_peaks=False)
        mean_df[col] = means
        amp_df[col] = amps
    
    # Correct amplitude using calibration and compute statistics
    print(means.mean())
    amp_df = amp_df/np.polyval(cal_poly, mean_df)
    # amp_df['avg'] = amp_df.mean(axis=1)
    # amp_df['stdev'] = amp_df.std(axis=1)
    # amp_df['rel_stdev'] = amp_df.stdev/amp_df.avg
    print(f'Mean Voltage: {mean_df.mean().mean()}')
    # Add avg to wirescan_df
    wirescan_df[offset] = amp_df.mean(axis=1)
    
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
    dz_p1_und = 48 #inches
    dz_und_p2 = 10.5 #inches
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






# %%
