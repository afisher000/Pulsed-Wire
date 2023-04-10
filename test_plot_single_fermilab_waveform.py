

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from utils_pulsedwire import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import os

folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 2'
filename = 'corrected xtraj.csv'

path = os.path.join(folder, filename)

def plot_file(path):
    signal = pd.read_csv(path, header=None)[0]
    N = len(signal)
    dt = 1e-7
    c0 = 250
    dz = .3*.032
    window = int(dz/(c0*dt))
    smooth = savgol_filter(signal, window, 5)
    time = np.arange(0, N*dt, dt)

    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.plot(time, smooth)
    return

plot_file(path)# %%

# %%
