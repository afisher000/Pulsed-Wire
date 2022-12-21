# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:31:21 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import rfft, rfftfreq, irfft

    
def get_signal_means_and_amplitudes(time, signal):
    # Smooth with savgol filter
    smooth_signal = savgol_filter(signal, len(signal)//500+1, 3)
    
    # Downsample for speed
    sample_points = 10000
    sample_rate = len(smooth_signal)//sample_points
    downsampled_signal = smooth_signal[::sample_rate]
    downsampled_time = time[::sample_rate]
    
    # Take numerical derivative, find peaks
    signal_derivative = np.append(0, np.diff(downsampled_signal))
    pk_idxs, _ = find_peaks(np.abs(signal_derivative), prominence = signal_derivative.max()*.5)

    # Find mean zero crossings of derivative between derivative peaks
    zero_idxs = []
    for j in range(len(pk_idxs)-1):
        scn = np.arange(pk_idxs[j], pk_idxs[j+1])
        sign = signal_derivative[pk_idxs[j]]
        zero_idx = np.logical_and(
            sign*signal_derivative[scn]>0, 
            sign*signal_derivative[scn+1]<=0
        ).nonzero()[0].mean()
        zero_idxs.append(scn[int(zero_idx)])
    
    
    # Search for signal peaks around derivative zero_crossings
    pk_times = []
    pk_vals = []
    pk_errors = []
    for idx in zero_idxs:
        scn = np.arange(idx-10, idx+10)
        poly = np.polyfit(downsampled_time[scn], downsampled_signal[scn], 2)
        signal_fit = np.polyval(poly, downsampled_time[scn])
        if poly[0]>0:
            pk_idx = np.argmin(signal_fit)
        else:
            pk_idx = np.argmax(signal_fit)
            
        pk_times.append(downsampled_time[scn[pk_idx]])
        pk_vals.append(signal_fit[pk_idx])
        pk_errors.append(signal_fit[pk_idx] - downsampled_signal[scn[pk_idx]])
    
    
    # Return amplitudes and means as computed from peaks
    amps    = np.abs(np.convolve(pk_vals, [0.5, -1, 0.5], mode='valid'))
    means   = np.convolve(pk_vals, [.25, .5, .25], mode='valid')
    
    # Apply correction for strong deflections in signal
    dy = means[2:]-means[:-2]       # change in volts over period
    delta = 2*dy/amps[1:-1]               # correction fit to relative error using delta parameter
    rel_amp_error = 3e-5*delta**4 + 1.27e-2*delta**2
    amps[1:-1] = amps[1:-1]*(1-rel_amp_error)
    
    # Drop first and last peak locations
    means = means[1:-1]
    amps = amps[1:-1]
    
    return means, amps

def low_pass_filter(time, signal, cutoff):
    yf = rfft(signal)
    freq = rfftfreq(len(time), time[1]-time[0])
    yf[freq>cutoff] = 0
    filtered_signal = irfft(yf, len(time))

    # Clip ends of filtered signal (show some anomaly)
    window = len(time)//50
    filtered_signal[:window] = filtered_signal[window]
    filtered_signal[-window:] = filtered_signal[-window]

    return filtered_signal

def get_fourier_transform(time, signal, freq_range=None, reduce_fmax=1, 
                          reduce_df=1):

    # Pad/skip data to change frequency resolution (df separation)
    pad_points = int((reduce_df-1)*len(signal)/2)
    data_values = np.pad(signal, 
                            (pad_points, pad_points),
                            mode='edge')

    # Reduce fmax to downsample and speed up computation
    reduce_fmax = int(reduce_fmax)
    downsampled_signal = data_values[::reduce_fmax]
    downsampled_time = time[::reduce_fmax]-time[0]

    # Take fourier transform
    yf = rfft(downsampled_signal)
    freq = rfftfreq(len(downsampled_signal), np.diff(downsampled_time).mean())

    # Scale spectra by npoints and df
    yf = yf/len(downsampled_signal)/np.diff(freq).mean()

    # Shift to t=0
    freq = np.real(freq)
    amp = np.real(abs(yf))
    phase = np.real(np.angle(yf))

    # Truncate results to freq_range
    if freq_range is not None:
        scn = np.logical_and(freq>freq_range[0], freq<freq_range[1])
        freq = freq[scn]
        amp = amp[scn]
        phase = phase[scn]

    return freq, amp, phase