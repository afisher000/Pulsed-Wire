# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:16:14 2022

@author: afish
"""
import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import scope 
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

ref = pd.read_csv('2022-07-18 earth_deflection.2.csv')
meas = ref.rename(columns={'y':'data'})
ref_amps, ref_means = pwf.get_measurement_amplitudes(meas, annotate_plot=False, ref_magnet=False, return_means=True)

data = pd.read_csv('2022-07-18 test signal.5.csv')
meas = data.rename(columns={'y':'data'})
amps, means = pwf.get_measurement_amplitudes(meas, annotate_plot=False, ref_magnet=False, return_means=True)

rel_error = 100*(amps.mean() - ref_amps.mean())/ref_amps.mean()
print(f'Difference in amplitudes: {rel_error:.1f}%')

# Qfit of reference
fitdata = ref[ref.time.between(0, 2.5e-3)]
qfit = np.polyfit(fitdata.time, fitdata.y, 3)

# Plot
ax = ref.plot(x='time', y='y')
ax.plot(data.time, np.polyval(qfit, data.time))
ax.plot(data.time, data.y)

# Subtract curvature from signal
data['fit'] = np.polyval(qfit, data.time)
fig, ax = plt.subplots()
ax.plot(data.time, data.y - data.fit)
ax.set_xlim([0, data.time.max()])