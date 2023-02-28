# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:41:07 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import utils_pulsedwire as up
import matplotlib.pyplot as plt
plt.close('all')


c0=250
EIwT = 6.4e-8
reduce_dt=5
reduce_fmax=5
reduce_df=2

period = .001
time = np.arange(-10*period, 10*period, period/100)
signal = np.sin(time*2*np.pi/period)*np.heaviside(-1*(time-period/2), 0)*np.heaviside((time+period/2), 0)
plt.plot(time, signal)

# time = 
# time = time.values if hasattr(time, 'values') else time
# signal = signal.values if hasattr(signal, 'values') else signal
    
# freq, yf, downsampled_time = get_fourier_transform(time, signal, reduce_fmax=reduce_fmax, reduce_df=reduce_df)

# # Compute frequency domain variables
# omega = 2*np.pi*freq
# if EIwT==0:
#     k = omega/c0
# else:
#     k = np.sqrt( np.sqrt(omega**2/c0**2/EIwT + 1/4/EIwT**2) - 1/2/EIwT)
# speed = c0*np.sqrt(1+EIwT*k**2)

# # Apply dispersion correction
# Fk = (speed/c0)**3 + EIwT * k**2 * speed/c0
# yf0 = (yf*Fk*np.exp(-1j*omega*downsampled_time[0])).reshape((-1,1)) 
# k0 = k
# dk0 = np.diff(k0, append=k0[-1]).reshape((-1,1))


# # Get signal by manual integration (k0 not unequally spaced)
# corrected_time = time[::reduce_dt]
# matrix = yf0 * np.exp(1j*np.outer(c0*k0, corrected_time)) * c0 * dk0/(2*np.pi)
# corrected_signal = (np.sum(matrix, axis=0) - 0.5*matrix[0,:]).real*2

