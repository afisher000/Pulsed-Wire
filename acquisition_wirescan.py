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
from utils_pulsedwire import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import os

# Folder and file
folder = '2023-06-05 xtraj, yoffset'
file = '(0,1500).csv'

# 
filename = os.path.join(folder, file)

# Take measurements
scope = Scope()
# filename = os.path.join(folder, 'ytraj', file)
# scope.get_measurements(channel=1, shots=20, validate='none', update_zero=False, npoints=10000)
# scope.save_measurements(filename)

# filename = os.path.join(folder, 'xtraj', file)
scope.get_measurements(channel=2, shots=20, validate='none', update_zero=False, npoints=10000)
# scope.save_measurements(filename)


# %%

# %%

