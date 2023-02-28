# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:52:28 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
plt.close('all')

# Create sinusoid
pi = 3.1415926535
dz = 1/1000
z = np.arange(-10, 10, dz)
sine = np.sin(z*2*pi)


def error_from_smoothing(zspan, porder=3):
    # Apply savgol filter
    points = int(zspan/dz) if int(zspan/dz)%2==1 else int(zspan/dz)+1
    filtered_sine = savgol_filter(sine, points, porder)
    
    # Errors
    rel_error = max(filtered_sine-sine)/max(sine)
    return rel_error

zspans = np.linspace(0.01,1,100)
rel_errors_3 = [error_from_smoothing(zspan, 3) for zspan in zspans]
rel_errors_5 = [error_from_smoothing(zspan, 5) for zspan in zspans]


fig, ax = plt.subplots()
ax.plot(zspans, rel_errors_3, label='Cubic')
ax.plot(zspans, rel_errors_5, label='Quintic')
ax.set_xlabel('Fraction of period')
ax.set_ylabel('Relative error')


