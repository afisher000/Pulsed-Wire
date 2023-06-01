# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:41:07 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import savgol_filter
plt.close('all')


def get_fourier_transform(time, signal, freq_range=None, reduce_fmax=1, 
                          reduce_df=1):

    # Pad/skip data to change frequency resolution (df separation)
    reduce_df = int(reduce_df)
    pad_points = int((reduce_df-1)/2*len(signal))
    pad_time = (reduce_df-1)/2*(time[-1]-time[0])
    padded_time = np.pad(time, pad_points, mode='linear_ramp', 
                         end_values=[time[0]-pad_time, time[-1]+pad_time]
    )
    padded_signal = np.pad(signal, pad_points, mode='edge')
    
    
    # Reduce fmax to downsample and speed up computation
    reduce_fmax = int(reduce_fmax)
    downsampled_signal = padded_signal[::reduce_fmax]
    downsampled_time = padded_time[::reduce_fmax]

    # Take fourier transform
    yf = rfft(downsampled_signal)
    freq = rfftfreq(len(downsampled_signal), np.diff(downsampled_time).mean())

    fig, ax = plt.subplots()
    ax.plot(freq, np.abs(yf))
    
    # Scale spectra by npoints and df
    yf = yf/len(downsampled_signal)/np.diff(freq).mean()
    freq = np.real(freq)


    # Truncate results to freq_range
    if freq_range is not None:
        scn = np.logical_and(freq>freq_range[0], freq<freq_range[1])
        freq = freq[scn]
        yf = yf[scn]


    return freq, yf, downsampled_time



c0=250
EIwT = 6.4e-12
reduce_dt=5
reduce_fmax=5
reduce_df=10
pulsewidth = 1e-8

period = .001
time = np.arange(-10*period, 10*period, period/100)
signal = np.sin(time*2*np.pi/period)*np.heaviside(-1*(time-period/2), 0)*np.heaviside((time+period/2), 0)
signal += np.tanh(time*1e8)



df = pd.read_csv('veronica test.csv')
time = df.time
signal = df.signal

time = time.values if hasattr(time, 'values') else time
signal = signal.values if hasattr(signal, 'values') else signal
signal = savgol_filter(signal, 21, 3)
    
freq, yf, downsampled_time = get_fourier_transform(time, signal, reduce_fmax=reduce_fmax, reduce_df=reduce_df)

# Compute frequency domain variables
omega = 2*np.pi*freq
if EIwT==0:
    k = omega/c0
else:
    k = np.sqrt( np.sqrt(omega**2/c0**2/EIwT + 1/4/EIwT**2) - 1/2/EIwT)
speed = c0*np.sqrt(1+EIwT*k**2)

# Apply dispersion correction
Fklong = (speed/c0)**3 + EIwT * k**2 * speed/c0
Fkshort = (speed/c0) * (speed/c0 + c0/speed*EIwT*k**2) * (1j*omega*pulsewidth) / (np.exp(1j*omega*pulsewidth)-1)
Fkshort[0] = 1

Fk = Fkshort
yf0 = (yf*Fk*np.exp(-1j*omega*downsampled_time[0])).reshape((-1,1)) 
k0 = k
dk0 = np.diff(k0, append=k0[-1]).reshape((-1,1))


# Get signal by manual integration (k0 not unequally spaced)
corrected_time = time[::reduce_dt]
matrix = yf0 * np.exp(1j*np.outer(c0*k0, corrected_time)) * c0 * dk0/(2*np.pi)
corrected_signal = (np.sum(matrix, axis=0) - 0.5*matrix[0,:]).real*2

fig, ax = plt.subplots()
ax.plot(time, signal)
ax.plot(corrected_time, corrected_signal)


