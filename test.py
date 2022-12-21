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


# Take initial measurement
scope = Scope()
scope.get_measurements(channel=2, shots=1, validate='clipping', update_zero=True)
time = scope.time
signal = scope.data[0,:]




plt.plot(time, signal)
plt.plot(time, filtered_signal)
plt.show()


# %%
