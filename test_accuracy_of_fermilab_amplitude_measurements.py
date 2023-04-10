

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from utils_pulsedwire import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import os

data_path = 'C:\\Users\\afish\\Documents\\Fermilab Measurements'

def test_get_amps(index):
    filename = f'test{index}.csv'
    path = os.path.join(data_path, filename)

    signal = pd.read_csv(path, header=None)[0]
    N = len(signal)
    dt = 1e-7
    c0 = 250
    dz = .3*.032
    window = int(dz/(c0*dt))
    smooth = savgol_filter(signal, window, 5)
    time = np.arange(0, N*dt, dt)

    means, amps = get_signal_means_and_amplitudes(time, smooth)
    return amps


Namps = 53
Nfiles = 5
amps_matrix = np.zeros((Nfiles, Namps))
for index in range(Nfiles):
    amps_matrix[index, :] = test_get_amps(index)
stds = amps_matrix.std(axis=0)
means = amps_matrix.mean(axis=0)



# %%
