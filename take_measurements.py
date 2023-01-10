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
import os

# Folder and file
folder = 'Breaksection Alignment'
file = 'aligned_0.csv'

# 
filename = os.path.join(folder, file)
# Take measurements
scope = Scope()
# scope.get_measurements(channel=1, shots=30, validate='manual', update_zero=False)
# scope.get_measurements(channel=1, shots=30, validate='none', update_zero=False)
# scope.save_measurements(filename)

scope.print_waveforms(channels=['x','y'])



# %%
