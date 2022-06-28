# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:50:51 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




''' INPUTS '''
path = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\April 2022 Pulse Wire\\Centering Wire'
#file = '2022-05-12 (ytraj, xoffset) Andrew.csv'
files = '2022-06-01 (ytraj, xoffset) Andrew.csv'

#path = ''
#file = '2022-06-02 (xtraj, yoffset).csv'

# Loop over all files
#files = []
#for file in os.listdir(path):
#    if file.endswith('.csv') and ('traj' in file) and ('offset' in file):
#        files.append(file)

''' CODE '''
if not isinstance(files, list):
    files = [files]
    
for file in files:
    plt.close('all')
    print(f'Analyzing file {file}')
    # Parse trajectory and offset direction
    traj = file[file.find('traj')-1]
    offset = file[file.find('offset')-1] + 'offset'
    
    # Munge dataframe 
    wirescan = pd.read_csv(os.path.join(path,file))
    wirescan = wirescan.set_index([offset, 'iteration']).sort_index(level=[0,1])
    wirescan.rename(columns={traj:'data'}, inplace=True)
    
    # Build dataframe of signal amplitudes for each measurement 
    arr = []
    for index in wirescan.index.unique():
        measurement = wirescan.loc[index]
        amplitudes = pwf.get_measurement_amplitudes(measurement, annotate_plot=False)
        arr.append(amplitudes)    
    scan_amps = pd.DataFrame(arr, index=wirescan.index.unique()).add_prefix('peak')
    
    # Apply quad fits to concavity, normalize amplitudes
    offsets = scan_amps.index.get_level_values(level=0)
    try:
        wire_position, qfit_extrema = pwf.fit_concavity(offsets, scan_amps.values)
    except np.linalg.LinAlgError:
        print('SVD did not converge. Skipping to next wirescan file.')
        continue
    scan_amps = scan_amps/qfit_extrema
    scan_amps.name = f'{traj}traj, {offset}'
    
    # Wire alignment
    fig_wire_alignment, ax = plt.subplots()
    ax.plot(wire_position)
    ax.set_xlabel('Peak Number')
    ax.set_ylabel(offset)
    
    # Concavity plots
    fig_concavity = pwf.plot_concavity(scan_amps, wire_position, range(4))
    
    # Save figures
    figure_folder = 'Pulsedwire Plots'
    if not os.path.exists(os.path.join(path,figure_folder)):
        os.mkdir(os.path.join(path, figure_folder))
        
    wire_alignment_jpg = 'wire alignment - ' + file.strip('.csv') + '.jpg'
    concavity_jpg = 'concavity - ' + file.strip('.csv') + '.jpg' 
    fig_wire_alignment.savefig(os.path.join(path, figure_folder, wire_alignment_jpg))
    fig_concavity.savefig(os.path.join(path, figure_folder, concavity_jpg))
