# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:40:56 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
plt.close('all')



# Read data
fourier_kwargs = dict(freq_range=[-1,4e6],
                      reduce_fmax = 100, reduce_df=30, unwind_phase=True)
column_map = {'y':'data'}

td = pd.read_csv('dispersion_10in.csv').rename(columns={'y':'data'})
td, fd = pwf.correct_dispersion(td, fourier_kwargs = fourier_kwargs,
                                reduce_tpoints = 100)

td2 = pd.read_csv('dispersion_0in.csv').rename(columns={'y':'data'})
td2, fd2 = pwf.correct_dispersion(td2, fourier_kwargs = fourier_kwargs,
                                  reduce_tpoints=100)

td.data = td.data0
td2.data = td2.data0

fd = pwf.get_fourier_transform(td, **fourier_kwargs)
fd2 = pwf.get_fourier_transform(td2, **fourier_kwargs)


dz = 10*.0254
velocity = 2*np.pi*fd.freq * dz / (fd2.phase-fd.phase)

fig, ax = plt.subplots()
ax.plot(fd.freq, velocity)
ax.set_xlim([0, 2e4])
# ax.set_ylim([207,211])

# fig, ax = plt.subplots()
# ax.plot(fd.phase)
# ax.plot(fd2.phase)

# Fit theory curve parameters
def EulerBernoulli(k, c0, EIwT):
    return c0*np.sqrt(1+EIwT*k**2)
(c0, EIwT), _ = curve_fit(EulerBernoulli, kvector, velocity, p0=[velocity.mean(), 0])