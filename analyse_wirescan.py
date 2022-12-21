# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:26:55 2022

@author: afisher
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulsedwire_functions_edited import get_signal_means_and_amplitudes
import seaborn as sns

folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive\\2022-12-19 xtraj, yoffset (2)'

traj_coord = folder[folder.find('traj')-1]
offset_coord = folder[folder.find('offset')-1] + 'offset'

# Get quartic fit of calibration curve
calibration = pd.read_csv(os.path.join(folder, 'calibration.csv'))
calibration = calibration.sort_values(by='voltage')
cal_poly = np.polyfit(calibration.voltage, calibration.amplitude, 4)

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
        means, amps = get_signal_means_and_amplitudes(time, signal, plot_signal_peaks=False, plot_derivative_peaks=False)
        mean_df[col] = means
        amp_df[col] = amps
    
    # Correct amplitude using calibration and compute statistics
    amp_df = amp_df/np.polyval(cal_poly, mean_df)
    # amp_df['avg'] = amp_df.mean(axis=1)
    # amp_df['stdev'] = amp_df.std(axis=1)
    # amp_df['rel_stdev'] = amp_df.stdev/amp_df.avg

    # Add avg to wirescan_df
    wirescan_df[offset] = amp_df.mean(axis=1)
    
wirescan_df = wirescan_df.transpose().sort_index()
    

def get_axis_fit(series):
    peak = series.name
    poly = np.polyfit(series.index, series.values, 2)
    axis = -poly[1]/(2*poly[0])
    residuals = series.values - np.polyval(poly, series.index)
    rmse = np.sqrt(np.sum(residuals**2))
    return peak, axis, rmse
    
concavity_df = wirescan_df.apply(get_axis_fit).transpose()
concavity_df.columns = ['peak','axis','rmse']
cmap = plt.get_cmap('coolwarm')
concavity_df.plot.scatter('peak','axis', c='rmse', cmap=cmap)


