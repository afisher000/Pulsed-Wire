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

    
def get_signal_means_and_amplitudes(time, signal, plot_signal_peaks=False, plot_derivative_peaks=False, return_peaks=False):
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
    
    # Handle too few or too many peaks
    if len(pk_idxs)<58:
        raise ValueError('Threshold for finding derivative peaks might need decreased.')
    elif len(pk_idxs)>58:
        # print('Keeping the 58 largest peaks.')
        pks = np.abs(signal_derivative[pk_idxs])
        argsorting = np.argsort(pks)
        pk_idxs = np.sort(pk_idxs[argsorting[-58:]])
        
    # Plot derivative peaks
    if plot_derivative_peaks:
        fig, ax = plt.subplots()
        ax.plot(downsampled_time, signal_derivative)
        ax.scatter(downsampled_time[pk_idxs], signal_derivative[pk_idxs], c='r')
        
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
        
    # Check peaks
    if plot_signal_peaks:
        fig, ax = plt.subplots()
        ax.plot(downsampled_time, downsampled_signal)
        ax.scatter(pk_times, pk_vals)
        
    if return_peaks:
        return pk_times, pk_vals
    
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

def low_pass_filter(time, signal, cutoff, dcutoff):
    '''Time and signal are input vectors for transform. Frequency filtering is
    achieved with a logistic function centered at cutoff with width of 
    dcutoff.
    Logistic width is defined as the (1-tanh(width))/2 = 0.995.'''
    
    # Pad to avoid edge effects
    dt = time[1]-time[0]
    padded_time = np.hstack([
        time + (time[0]-time[-1]) - dt,
        time,
        time + (time[-1]-time[0]) + dt
    ])
    padded_signal = np.pad(signal, (len(signal),len(signal)), mode='edge')
    
    # Take fourier transform
    yf = rfft(padded_signal)
    freq = rfftfreq(len(padded_time), dt)
    
    # Apply cutoff with logistic function
    scaling = (1-np.tanh(3/dcutoff*(freq-cutoff)))/2
    yf *= scaling
    # yf[freq>cutoff] = 0 #Hard edge
    
    # Take inverse transform
    filtered_padded_signal = irfft(yf, len(padded_time))
    filtered_signal = filtered_padded_signal[len(signal):2*len(signal)]
    return filtered_signal

def get_fourier_transform(time, signal, freq_range=None, reduce_fmax=1, 
                          reduce_df=1):

    # Pad/skip data to change frequency resolution (df separation)
    pad_points = int((reduce_df-1)*len(signal)/2)
    data_values = np.pad(signal, (pad_points, pad_points), mode='edge')

    # Reduce fmax to downsample and speed up computation
    reduce_fmax = int(reduce_fmax)
    downsampled_signal = data_values[::reduce_fmax]
    downsampled_time = time[::reduce_fmax]-time[0]

    # Take fourier transform
    yf = rfft(downsampled_signal)
    freq = rfftfreq(len(downsampled_signal), np.diff(downsampled_time).mean())

    # Scale spectra by npoints and df
    yf = yf/len(downsampled_signal)/np.diff(freq).mean()
    freq = np.real(freq)


    # Truncate results to freq_range
    if freq_range is not None:
        scn = np.logical_and(freq>freq_range[0], freq<freq_range[1])
        freq = freq[scn]
        yf = yf[scn]


    return freq, yf

def correct_dispersion(time, signal, c0=207, EIwT = 2*6.4e-8, reduce_dt=10, reduce_fmax=20):
    freq, yf = get_fourier_transform(time, signal, reduce_fmax=reduce_fmax)
    # Take fourier transform, computer over values
    # freq = rfftfreq(len(time), time[1]-time[0])
    # yf = rfft(signal)
    
    # Compute frequency domain variables
    omega = 2*np.pi*freq
    if EIwT==0:
        k = omega/c0
    else:
        k = np.sqrt( np.sqrt(omega**2/c0**2/EIwT + 1/4/EIwT**2) - 1/2/EIwT)
    speed = c0*np.sqrt(1+EIwT*k**2)
    
    # Apply dispersion correction
    Fk = (speed/c0)**3 + EIwT * k**2 * speed/c0
    yf0 = (yf*Fk*np.exp(-1j*omega*time[0])).reshape((-1,1)) 
    k0 = k
    dk0 = np.diff(k0, append=k0[-1]).reshape((-1,1))

    
    # Get signal by manual integration (k0 not unequally spaced)
    corrected_time = time[::reduce_dt]
    matrix = yf0 * np.exp(1j*np.outer(c0*k0, corrected_time)) * c0 * dk0/(2*np.pi)
    corrected_signal = (np.sum(matrix, axis=0) - 0.5*matrix[0,:]).real*2
    
    
    return corrected_time, corrected_signal

def get_velocity_from_trajectory(time, signal, fix_dispersion=False, c0=207, EIwT=2*6.4e-8):
    if fix_dispersion:
        time, signal = correct_dispersion(time, signal, c0=c0, EIwT=EIwT)
        
    trajectory = low_pass_filter(time, signal, 2e4, 1e3)
    
    # Get velocity
    velocity = np.diff(trajectory, prepend=trajectory[0])
    filtered_velocity = low_pass_filter(time, velocity, 2e4, 1e3)
    return filtered_velocity
    
def get_field_from_trajectory(time, signal, fix_dispersion=False, c0=207, EIwT = 2*6.4e-8):
    filtered_velocity = get_velocity_from_trajectory(
        time, signal, fix_dispersion=fix_dispersion, c0=c0, EIwT=EIwT
    )
    field = np.diff(filtered_velocity, prepend=filtered_velocity[0])
    return time, field