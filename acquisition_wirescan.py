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
from utils_pulsedwire_edited import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import os

# Folder and file
folder = '2023-01-20 ytraj, xoffset'
file = '(500,0).1.csv'

# 
filename = os.path.join(folder, file)
# Take measurements
scope = Scope()
scope.get_measurements(channel=2, shots=30, validate='manual', update_zero=False)
# scope.get_measurements(channel=1, shots=30, validate='none', update_zero=False)
# scope.get_measurements(channel=1, shots=1, npoints=10000)
scope.save_measurements(filename)

# scope.print_waveforms(channels=['x','y'], npoints=10000, filename = filename)




# %%
