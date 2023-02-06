# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:58:02 2022

@author: afisher
"""


import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import scipy.optimize
import numpy as np
import pulsedwire as pwf
import matplotlib.pyplot as plt
import pandas as pd
plt.close('all')

# Might need to use linear ramp with signal?


file = '2022-07-12 (ytraj, xoffset).csv'
td = pd.read_csv(file)
td = td[td.xoffset==0].drop(columns=['xoffset','iteration']).rename(columns={'y':'data'}).reset_index(drop=True)

fourier_kwargs = dict(freq_range=[-1, 8e4], 
                      reduce_fmax=100, 
                      reduce_df=30,
                      unwind_phase=False)
td, fd = pwf.correct_dispersion(td, fourier_kwargs=fourier_kwargs, reduce_tpoints = 100)

fd.plot(x='freq', y='amp', marker='*', ylim=[0, fd.amp[1]], xlim=[0, 10000])


meas = td[['time','data']].copy()
amps1, means1 = pwf.get_measurement_amplitudes(meas,
                                        annotate_plot=False,
                                        ref_magnet=False,
                                        return_means=True)

meas = td[['time','data0']].rename(columns={'data0':'data'}).copy()
amps2, means2 = pwf.get_measurement_amplitudes(meas,
                                    annotate_plot=False,
                                    ref_magnet=False,
                                    return_means=True)    

fig, ax = plt.subplots()
ax.plot(range(len(amps1)), amps1, label='Dispersion')
ax.plot(range(len(amps2)), amps2, label='No Dispersion')
ax.set_ylabel('Amplitude of Peaks')
ax.set_xlabel('Peak Number')
ax.legend()
        