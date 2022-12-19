# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:19:29 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from pulsedwire_functions_edited import get_signal_means_and_amplitudes

    
shot_data = pickle.load(open('data.pkl', 'rb'))
time = pickle.load(open('time.pkl', 'rb'))

for j in range(shot_data.shape[0]):
    signal = shot_data[j, :]
    means, amps = get_signal_means_and_amplitudes(time, signal)
    print(f'Shot {j}: \n\tMean = {means.mean():.2f}mV\n\tAmplitude = {amps.mean()*1000:.1f}mV')






