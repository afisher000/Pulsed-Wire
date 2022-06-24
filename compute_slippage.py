# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:29:22 2022

@author: afish
"""

import scope_functions as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

plt.close('all')
def plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    return

def get_signal_region(df):
    # Look at body of signal
    full_amp = df.volts.max() - df.volts.min()
    peaks = np.sort(np.append(find_peaks(df.volts, prominence=full_amp/2)[0],
                              find_peaks(-1*df.volts, prominence=full_amp/2)[0]))
    signal = df.loc[peaks.min():peaks.max(), ['time','volts']]
    return signal

def integrate_velocity(df, xinc, window=201, porder=5):
    df['savgol'] = savgol_filter(df.volts, window, porder)
    df['traj'] = df.savgol.cumsum()/xinc 
    return

wfm = pd.Series({'xinc':2e-7, 'ymult':4e-3, 'yzero':3.7884, 'points':10000})
# Read in and clean data
fig_v, ax_v = plt.subplots()
fig_t, ax_t = plt.subplots()
for thresh_fac in [1, 1.5, 2]:
    yvel = pd.read_csv('xvelocity_5us.csv')
    yvel = sf.remove_noisy_shots(yvel, thresh_fac=thresh_fac)
    
    yvel['volts'] = yvel.volts - yvel.volts.mean()
    ax_v.plot(yvel.time, yvel.volts, label=f'thresh={thresh_fac}')
    
    yvel = get_signal_region(yvel)
    integrate_velocity(yvel, wfm.xinc)
    ax_t.plot(yvel.time, yvel.traj, label=f'thresh={thresh_fac}')

ax_v.set_title('Average of "clean" velocity shots')
ax_t.set_title('Integrating a savgol fit')
ax_v.legend()
ax_t.legend()

