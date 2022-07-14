# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:26:42 2022

@author: afisher
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

radius = 25e-6 #m
density = 8.25/1000*1e6 #kg/m^3
tension = 1 #N
EIw = 2e-7 #N*m^2 (from paper)
linear_density = density * (np.pi*radius**2)

def get_fourier_transform(time_data, max_freq=None):
    td = time_data
    yf = rfft(td.y.values)
    freq = rfftfreq(len(td), td.time.diff().mean())
    amp = abs(yf)
    phase = np.angle(yf)
    phase = (phase - 2*np.pi*freq*td.time.min())%(2*np.pi) #apply phase from time shift
    
    fourier_data = pd.DataFrame(np.vstack([freq, amp, phase]).T, columns=['freq','amp','phase'])
    if max_freq is None:
        return fourier_data
    else:
        return fourier_data[fourier_data.freq<max_freq]

def plot_fourier_transform(fourier_data, axes=None):
    fd = fourier_data
    if axes is None:
        fig, axes = plt.subplots(nrows=2, sharex=True)
        
    axes[0].plot(fd.freq,fd.amp)
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(fd.freq, fd.phase)
    axes[1].set_ylabel('Phase')
    axes[1].set_xlabel('Frequency (1/s)')
    axes[0].set_xlim([0,4e4])
    axes[0].set_ylim([0, fd.amp[1]])
    return axes


plt.close('all')
td10 = pd.read_csv('dispersion_5in.csv')
fd10 = get_fourier_transform(td10, max_freq = 1e4)
td0 = pd.read_csv('dispersion_0in.csv')
fd0 = get_fourier_transform(td0, max_freq = 1e4)

freq = fd0.freq
dz = 5*.0254
dphase = -(fd10.phase - fd0.phase)%(2*np.pi)

# Make phase monotonically increasing (assume no jump in dstep > 2*pi)
cycles = np.cumsum(np.diff(dphase)<0)
dphase[1:] = dphase[1:] + cycles* (2*np.pi)
velocity = freq * dz / dphase

fig, ax = plt.subplots()
ax.plot(freq, velocity)
ax.set_xlabel('Frequency (1/s)')
ax.set_ylabel('Speed (m/s)')


fig, ax = plt.subplots()
ax.plot(fd10.freq, fd10.phase)
ax.plot(fd0.freq, fd0.phase)
ax.plot(fd0.freq, dphase)
ax.set_xlim([0,4e4])













