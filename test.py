# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:26:55 2022

@author: afisher
"""
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulsedwire_functions_edited import get_signal_means_and_amplitudes
from oscilloscope_functions import Scope
import seaborn as sns

df = pd.read_csv('deduce_kick_in_velocity.csv')
time = df.time.values
up500 = df.up500.values
down1000 = df.down1000.values
_, amps_up = get_signal_means_and_amplitudes(time, up500, plot_derivative_peaks =False)
_, amps_down = get_signal_means_and_amplitudes(time, down1000)

#
t0 = 4e-3
dt = 5e-4

# Scale to same vertical
up500_scaled = up500
down1000_scaled = down1000 * amps_up.mean()/amps_down.mean()
peaks = np.arange(len(amps_down))

difference = up500_scaled-up500_scaled.mean() - (down1000_scaled-down1000_scaled.mean())
plt.plot(time, difference)

plt.xlim([t0-dt, t0+dt])
plt.show()
# %%
