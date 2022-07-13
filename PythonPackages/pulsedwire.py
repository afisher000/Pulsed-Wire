# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:51:47 2022

@author: afisher
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

def analyze_wirescan(file, path='', plot=True):
    print(f'Reading {file}')
    # Parse trajectory and offset direction
    traj = file[file.find('traj')-1]
    offset = file[file.find('offset')-1] + 'offset'
    
    # Munge wirescan dataframe
    wirescan = pd.read_csv(os.path.join(path,file))
    wirescan = wirescan.set_index([offset, 'iteration']).sort_index(level=[0,1])
    wirescan.rename(columns={traj:'data'}, inplace=True)
    
    # Create data DataFrame:
        # Columns = amplitudes, means, offsets
        # Index = Peak number (repeats for each offset)
    data = pd.DataFrame(columns=['amps','means','offset'])
    for meas_offset in wirescan.index.get_level_values(level=0).unique():
        meas = wirescan.loc[meas_offset].copy()
        meas.rename(columns = {traj:'data'}, inplace=True)
        amplitudes, means = get_measurement_amplitudes(meas,
                                                           annotate_plot=True,
                                                           ref_magnet=False,
                                                           return_means=True)
        temp_df = pd.DataFrame(np.array([amplitudes, 
                                         means, 
                                         meas_offset*np.ones_like(means)
                                         ]).T, 
                               columns=['amps','means','offset'])
        data = pd.concat([data, temp_df])
        
    
    # Create peak_data DataFrame:
        # Columns = axis, extrema (of quadratic fits)
        # Index = Peak number (no repeating)
    peak_data = pd.DataFrame(columns=['axis','extrema'])
    for peak in data.index.unique():
        peak_data.loc[peak] = fit_concavity(data.loc[peak].offset, data.loc[peak].amps)
    peak_data = peak_data.reset_index().rename(columns={'index':'peak'})
    
    
    
    if plot:
        # FIGURES 
        cmap = plt.get_cmap('coolwarm')
        
        # Plot amplitudes vs means
        data.plot.scatter(x='means', y='amps', c='offset', cmap=cmap) 
        
        # Plot amplitudes vs concavity
        data.plot.scatter(x='offset', y='amps', c = data.index, cmap=cmap)
        
        # Plot wire alignment
        peak_data.plot.scatter(x='peak', y='axis')

    return data, peak_data


def analyze_laserscan(laserscan, datacolumn, title=''):
    results = pd.DataFrame(columns=['zero','amp','amp_std'])
    for dist in laserscan.index.unique():
        data = laserscan.loc[dist].rename(columns = {datacolumn:'data'})
        amps = get_measurement_amplitudes(data, ref_magnet=False)*1000 #mV
        zero = data.data.iloc[0]*1000 #mV
        results.loc[dist] = [zero, amps.mean(), amps.std()]
        
    ax_calibration = results.reset_index().plot(x='index', y='zero', kind='scatter',
                               xlabel = 'Displacement (um)',
                               ylabel = 'Signal zero (mV)',
                               title = title)
    
    ax_amp = results.reset_index().plot(x='index', y='amp', kind='scatter',
                               xlabel = 'Displacement (um)',
                               ylabel = 'Mean Signal Amplitude (mV)',
                               title = title)
    
    # Apply linear fit
    fit_data = results[results.amp>results.amp.max()*.8]
    linefit = np.polyfit(fit_data.index, fit_data.zero,1)
    calibration = abs(linefit[0])
    ax_calibration.plot(fit_data.index, np.polyval(linefit, fit_data.index),
                        label=f'Calibration={calibration:.1f} mV/um')
    ax_calibration.legend()
    
    # Compute deflection
    deflection = results.amp.max()/calibration/2
    return deflection

def get_linear_calibration(file, plot=False, ax=None):
    ''' Returns the linear calibration computed from datapoints in a file.
    The file must have the displacement column labeled 'dist'. '''
    
    data = pd.read_csv(file)
    assert('dist' in data.columns)
    
    cals = []
    for column in data.columns.drop('dist'):
        fit = np.polyfit(data.dist, data[column], 1)
        cal = np.abs(fit[0]*1000)
        cals.append(cal)
        
        if plot:
            ax = data.plot(x='dist', y=column, kind='scatter')
            ax.plot(data.dist, np.polyval(fit, data.dist), 
                    label=f'Fit: {cal:.1f} mV/um')
            ax.set_xlabel('Displacement (um)')
            ax.set_ylabel(column)
            ax.legend()
    
    return cals

def get_measurement_amplitudes(measurement, annotate_plot=False, ref_magnet=True, return_means=False):
    ''' Computes the relative amplitudes of the measurement.'''
    
    # Add savgol_filter smoothing for downsampling
    measurement['savgol'] = savgol_filter(measurement.data, round(len(measurement)/100+1), 3)
    
    # Meas is a downsampled version of measurement for quick calculations
    sample_rate = round(len(measurement)/1000)
    meas = measurement.iloc[::sample_rate].copy()
    meas = meas.reset_index(drop=True).drop(columns='data').rename(columns={'savgol':'data'})
    meas['datap'] = np.append(0, np.diff(meas.data))
    f_meas = interp1d(meas.time, meas.data) #Interpolation function for plots
        
    # Use derivate (datap) identify reference magnet and undulator signal
    pktimes, period = get_measurement_pktimes_from_derivative(meas.time, meas.datap)
    
    if annotate_plot:
        _, annotate = plt.subplots()
        annotate.plot(meas.time, meas.data, c='k')
        signal_range = meas.time.between(pktimes[0], pktimes[-1])
        annotate.plot(meas.time[signal_range], meas.data[signal_range])
        annotate.scatter(pktimes, f_meas(pktimes), c='r')
    else:
        annotate=None
    
    # Get reference amplitude
    ref_amp = 1
    if ref_magnet:
        ref = meas[meas.time<pktimes[0]-period]
        ref_amp = get_measurement_ref_amplitude(ref, window=period/2, annotate=annotate)
    
    # Compute means and amplitudes (peak to peak) of undulator signal
    peaks = [polyfit_peak(measurement.time, measurement.data, pktime, window=period/5)[1] 
             for pktime in pktimes]
    amplitudes = np.abs(np.convolve(peaks, [0.5, -1, 0.5], mode='valid'))[1:-1]
    means   = np.convolve(peaks, [.25, .5, .25], mode='valid')
    
    # Apply measurement correction for strong signal deflection
    deltay = means[2:]-means[:-2] # change in volts over period
    delta = 2*deltay/amplitudes # delta is fitting parameter of correction
    meas_error = 3e-5*delta**4 + 1.27e-2*delta**2 #correction fit to relative error
    amplitudes = amplitudes*(1-meas_error)
    
    if return_means:
        return amplitudes/ref_amp, means[1:-1]
    else:
        return amplitudes/ref_amp


def get_measurement_pktimes_from_derivative(time, datap):
    ''' Compute peaks and pktimes of measurement derivative. Return estimate
    peaks times for raw measurement.'''
        
    max_width = len(time)/10
    peaks, _ = find_peaks(datap, prominence=datap.max(), width=(0,max_width))
    pktimes = time[peaks].values
    
    period = np.diff(pktimes).mean()
    # If period variation is large, incorrect peaks were found...
    if len(peaks)==10:
        peaks = peaks[1:]
    
    max_pktimes = np.append(pktimes[0]-3/4*period, pktimes+period/4)
    min_pktimes = pktimes-period/4
    all_pktimes = np.sort(np.append(max_pktimes, min_pktimes))
    
    if len(peaks)!=9:
        print(f'Incorrect number of peaks: found {len(peaks)}')
        fig, ax = plt.subplots()
        ax.plot(time, datap)
        ax.scatter(time[peaks], datap[peaks])
        
    return all_pktimes, period

def get_measurement_ref_amplitude(ref, window=.08e-3, annotate=None):
    ''' Compute the reference amplitude as the difference in signal at two 
    points located between the first datapoint and magnet and the magnet and 
    first peak.'''
    weight = .3
    magnet_time = ref.time[ref.datap.abs().argmax()]
    ref1_time = magnet_time*weight + ref.time.values[0]*(1-weight)
    ref2_time = magnet_time*weight + ref.time.values[-1]*(1-weight)
    
    time1 = ref.time[ref.time.between(ref1_time-window, ref1_time+window)]
    time2 = ref.time[ref.time.between(ref2_time-window, ref2_time+window)]
    data1 = ref.data[ref.time.between(ref1_time-window, ref1_time+window)]
    data2 = ref.data[ref.time.between(ref2_time-window, ref2_time+window)]
    
    if annotate is not None:
        annotate.plot(time1, data1, c='g')
        annotate.plot(time2, data2, c='g')
    
    ref1 = data1.mean()
    ref2 = data2.mean()
    return abs(ref2-ref1)

def polyfit_peak(time, data, est_pktime, window=3e-5, porder=5, plot=False):
    ''' Fit the measurement data in a window about the estimated peak time to
    compute the exact peak time and value. '''
    pkdata = pd.DataFrame()
    pkdata['time'] = time[time.between(est_pktime-window, est_pktime+window)]
    pkdata['data'] = data[time.between(est_pktime-window, est_pktime+window)]
    pkdata['time_demeaned'] = pkdata.time - pkdata.time.mean()

    poly = np.polyfit(pkdata.time_demeaned, pkdata.data, porder)
    polydata = np.polyval(poly, pkdata.time_demeaned)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(pkdata.time, pkdata.data)
        ax.plot(pkdata.time, polydata)
        
    if poly[3]>0: #Check quadratic sign
        pktime = pkdata.time.iloc[polydata.argmin()]
        if polydata.min() in polydata[[0,-1]]:
            print('Extrema at edge of polyfit window!')
        return pktime, polydata.min()
    else:
        pktime = pkdata.time.iloc[polydata.argmax()]
        if polydata.max() in polydata[[0,-1]]:
            print('Extrema at edge of polyfit window!')
        return pktime, polydata.max()


def fit_concavity(offsets, amplitudes):
    '''Return the axis and offset of a poly fit'''
    a,b,c = np.polyfit(offsets, amplitudes, 2)
    axis = -b/(2*a)
    extrema = c-b**2/(4*a)
    return axis, extrema


def plot_concavity(scan_amps, wire_position, idxs=range(3), plot_theory=True):
    ''' Plot the concavities of specified peaks, along with theory optionally.'''
    traj_label, offset_label = scan_amps.name.title().split(', ')
    
    fig, ax = plt.subplots()
    offsets = scan_amps.index.get_level_values(level=0)
    for idx in idxs:
        ax.scatter(offsets, scan_amps.iloc[:,idx], label=f'Peak {idx}')
    ax.legend()
    ax.set_xlabel(f'{offset_label} (um)')
    ax.set_ylabel('Normalized Amp.')
    
    if idxs[0]>=0:
        ax.set_title(f'{traj_label}: Entrance Peaks')
    else:
        ax.set_title(f'{traj_label}: Exit Peaks')
            
    if plot_theory:
        theory_offsets = np.linspace(-1500,1500,100)
        ku = 2*np.pi/32e3 #um
        avg_wire_position = wire_position[idxs].mean()
        if ('Xtraj'==traj_label)!=('Xoffset'==offset_label):
            coeff = 1.53
            sign = 1
        else:
            coeff = 1.16
            sign = -1

        theory_amplitudes = 1 + sign*0.5*(ku*coeff*(theory_offsets-avg_wire_position))**2
        ax.plot(theory_offsets, theory_amplitudes, label='Theory')

    return fig



def merge_wirescan_files(folder, path='', channels={}, skiprows=0):
    ''' Combine wirescan files from a given folder (format: %Y-%m-%d (?trax, ?offset) )
    into a single csv file saved with the folder name. Specify a dictionary 
    'channels' to rename the dataframe columns to ['time','x','y'] as
    appropriate. Use skiprows to remove scope csv headers.'''
    
    '''EXAMPLE CODE FOR THUMBDRIVE DATA:
    channels = {'TIME':'time', 'CH1':'x', 'CH4':'y'}
    path = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\April 2022 Pulse Wire\\Centering Wire'
    
    for obj in os.listdir(path):
        # Skip non-folders
        if not os.path.isdir(os.path.join(path, obj)):
            continue
        
        # Only use files with traj and offset named
        if ('traj' in obj) and ('offset' in obj):
            merge_thumbdrive_wirescan_files(obj, path=path, channels=channels, skiprows=20)
    '''
    
    
    # Get trajectory and offset direction
    traj_coord = folder[folder.find('traj')-1]
    offset_coord = folder[folder.find('offset')-1]
    
    # 
    print(f'Gathering files from {folder}')
    dfs = []
    for file in os.listdir(os.path.join(path, folder)):
        data_string = file[:-4].replace('(','').replace(')','').replace(',','.')
        xoffset, yoffset, iteration = map(int, data_string.split('.'))
        
        df = pd.read_csv(os.path.join(path, folder, file), skiprows=skiprows)
        
        # If channels dictionary provided, rename
        if len(channels)>0:
            df.rename(columns=channels, inplace=True)
            if traj_coord == 'x':
                df.drop(columns='y', inplace=True)
            elif traj_coord == 'y':
                df.drop(columns='x', inplace=True)
        
        
        if offset_coord == 'x':
            df['iteration'] = iteration
            df['xoffset'] = xoffset
            df.set_index(['xoffset','iteration'], inplace=True)
        elif offset_coord == 'y': 
            df['iteration'] = iteration
            df['yoffset'] = yoffset
            df.set_index(['yoffset','iteration'], inplace=True)
    
        dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_csv(os.path.join(path, folder+'.csv'))
    print(f'Successfully merged into {folder}.csv')
    return