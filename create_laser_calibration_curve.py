# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:19:29 2022

@author: afisher
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from pulsedwire_functions_edited import get_signal_means_and_amplitudes, low_pass_filter
from oscilloscope_functions import Scope

def get_measurement(scope):
    scope.get_measurements(channel=1, shots=1, validate='clipping', update_zero=True)

# Setup figure
plt.ion()
fig, ax = plt.subplots()

# Take initial measurement
scope = Scope()
get_measurement(scope)

# Loop over measurements
avg_means = []
avg_amps = []
while True:
    # Average mean and amplitude
    time = scope.time
    signal = scope.data[0,:]

    # Filter high frequency noise
    filtered_signal = low_pass_filter(time, signal, 4e4)

    try:
        means, amps = get_signal_means_and_amplitudes(time, filtered_signal)
    except:
        print('Error. Try increasing signal/noise ratio...')
        get_measurement(scope)
        continue

    # Update plot
    avg_means.append(means.mean())
    avg_amps.append(amps.mean())
    ax.clear()
    ax.scatter(avg_means, avg_amps)
    fig.canvas.draw_idle()
    plt.pause(.05)

    # Save data every loop
    df = pd.DataFrame(np.vstack([avg_means, avg_amps]).T, columns=['voltage','amplitude'])
    df.to_csv('calibration.csv', index=False)

    # Get next measurement
    get_measurement(scope)


def clean_calibration_data():
    df = pd.read_csv('calibration.csv')
    df = df[df.amplitude>df.amplitude.max()*.7]
    df.to_csv('calibration.csv')
    return







# %%
