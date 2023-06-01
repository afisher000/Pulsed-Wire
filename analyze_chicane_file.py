
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, savgol_filter
from utils_pulsedwire import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import os

plt.close('all')

def read_zx_from_radia_file(file, folder='radia chicane text files', z0=0.678):
    path = os.path.join(folder, file)
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['z','x']
    
    # Convert to meters
    z = df.z/1000+z0
    xwire = df.x*1000
    
    # Scale xwire to x
    x = xwire * 0.058 / (xwire[0]-xwire.min())
    x = x-x[0]
    return z, x
    
    
    

def read_zx_from_file(file, folder='chicane pre-final shimming'):
    # I used xhat is to right, yhat is up
    path = os.path.join(folder, file)
    df = pd.read_csv(path)

    time = df.time.values
    signal = df.drop(columns=['time']).mean(axis=1)

    npoints = len(time)
    window = int(npoints/80)
    smoothed = savgol_filter(signal, window, polyorder=3)

    # Convert time to z and volts to x
    c0 = 0.510*2/.004 #travels 51cm twice in 4ms
    z = (time-time[0])*c0
    x = smoothed * 0.058 / (smoothed[0]-min(smoothed)) #max deflection is 58 mm in radia
    x = x-x[0]
    return z, x

def get_angle_fits(z, x, print_values=False):
    # Compute angular fits
    z0 = .4
    poly0 = np.polyfit(z[z<z0], x[z<z0], deg=1)
    ai = poly0[0]*1e3

    z1 = .85
    poly1 = np.polyfit(z[z>z1], x[z>z1], deg=1)
    af = poly1[0]*1e3
    if print_values:
        print(f'Initial angle = {ai:.1f} mrad')
        print(f'Final angle = {af:.1f} mrad')
    return ai, af

def plot_files(files, title):
    fig, ax = plt.subplots()
    for file in files:
        try:
            z, x = read_zx_from_file(file)
        except:
            z, x = read_zx_from_radia_file(file)         
            
        ai, af = get_angle_fits(z, x)
        label = f'{file}, {af-ai:.1f}mrad'
        ax.plot(z, x, label=label)

    ax.legend()
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('X (m)')
    ax.set_title(title)
    return

# Compare with radia
files = ['(0,0).csv','(0,0).txt']
plot_files(files, title='Comparison')

# # Consistency check for nominal wire position
# files = [
#     '(0,0).csv',
#     '(0,0).2.csv',
#     '(0,0).3.csv',
#     '(0,0).4.csv',
# ]
# plot_files(files, title='Reproducibility at nominal')

# # Variation in y
# files = [
#     '(0,-2).csv',
#     '(0,0).csv',
#     '(0,2).csv'
# ]
# plot_files(files, title='Vary ypos of wire')

# # Variation in x
files = [
    '(0,0).csv',
    '(0,0).txt',
    '(2,0).csv',
    '(2,0).txt'
]
plot_files(files, title='Vary xpos of wire')

# # Variation in shim
# files = [
#     '(0,0) no shims.csv',
#     '(0,0) 1shims all in.csv',
#     '(0,0) 2shims all in.csv',
#     '(0,0) 3shims all in.csv'
# ]
# plot_files(files, title='Vary shimming')

# # Final tuned
# files = ['final_tuned.csv']
# plot_files(files, title='Final tuning')

# %%
