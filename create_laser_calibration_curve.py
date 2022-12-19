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
from pulsedwire_functions_edited import get_signal_means_and_amplitudes
from oscilloscope_functions import Scope


def get_measurement(scope):
    scope.get_measurements(channel=2, shots=1, validate=True, update_zero=True)

# Setup figure
plt.ion()
fig, ax = plt.subplots()

# Take initial measurement
scope = Scope()
get_measurement(scope)

# Loop over measurements
avg_means = []
avg_amps = []
while scope.input != 'q':
    # Compute aggregate of next measurement
    time = scope.time
    signal = scope.data[0,:]
    try:
        means, amps = get_signal_means_and_amplitudes(time, signal)
    except:
        print('Error! Go to larger amplitude...')
        get_measurement(scope)
        continue

    # Update plot
    avg_means.append(means.mean())
    avg_amps.append(amps.mean())
    ax.clear()
    ax.scatter(avg_means, avg_amps)
    fig.canvas.draw_idle()

    # Save data
    pickle.dump(avg_means, open('means.pkl','wb'))
    pickle.dump(avg_amps, open('amps.pkl', 'wb'))
    
    # Get next measurement
    get_measurement(scope)






# %%
