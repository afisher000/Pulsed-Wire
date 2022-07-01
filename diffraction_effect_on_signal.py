# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:06:46 2022

@author: afisher
"""

import numpy as np
import matplotlib.pyplot as plt

# All distances in um
D = 1000 #slit to wire
slit = 50 #slit diameter
wire = 50 #wire diameter
wavelen = 0.7 #red laser


# Intensity distribution at wire location
coeff = np.pi*slit/wavelen/D
I0 = slit/wavelen/D #so int(I)=1

x = np.linspace(-100,100,100000)
dx = x[1]-x[0]
I_kernel = I0*(np.sin(coeff*x)/(coeff*x))**2
slit_window = round(slit/dx)
I = np.convolve(I_kernel, np.ones(slit_window)/slit_window, 'same')

# Check integral = 1
print(f'Integral={dx*I.sum()}')

# Convolve for summations
window = round(wire/dx)
blocked = np.convolve(I, dx*np.ones(window), 'same')
signal = 1-blocked
derivative = np.append(0, np.diff(signal))
derivative = derivative/derivative.max()

# Plot
plt.close('all')
fig, axes = plt.subplots(nrows=3, sharex=True)
axes[0].plot(x, I)
axes[0].set_ylabel('Norm. Intensity')

axes[1].plot(x, signal)
axes[1].set_ylabel('Norm. Signal')


axes[2].plot(x, derivative)
axes[2].set_ylabel('Norm. Derivative')
axes[2].set_xlabel('Wire Position (um)')
axes[2].set_xlim([-100,100])

# Plot linear region
mask = derivative>0.8
fig, ax = plt.subplots()
ax.plot(x[mask], derivative[mask])
ax.set_title('Linear Region')
ax.set_xlabel('Wire Position (um)')
ax.set_ylabel('Norm. Derivative')