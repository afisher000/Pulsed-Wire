# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:06:46 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def singleslit_diffraction(x, wavelen=0.7, aperature=50, D=1000):
    ''' Compute the intensity distribution of light (given wavelength)
    diffracting from a slit aperature (given diameter) propogated a distance D.'''
    coeff = np.pi*aperature/wavelen/D
    Icoeff = aperature/wavelen/D * aperature #Scale for laser intensity of 1

    # Compute diffraction from point source
    dx = x[1]-x[0]
    I_kernel = Icoeff*(np.sin(coeff*x)/(coeff*x))**2
    
    # Convolve against aperature
    window = round(aperature/dx)
    I = np.convolve(I_kernel, np.ones(window)/window, 'same')
    return I

def signal_slitafterwire(slit=50, wire=50, D=1000):
    ''' Compute signal of photodiode with the slit after the wire.'''
    x_wire = np.linspace(-100,100,173)
    x_slit = np.linspace(-3*slit,3*slit,10000) #several times wider than slit so convolution over aperature isn't truncated
    signal = np.zeros_like(x_wire)
    for j in range(len(x_wire)):
        I = 1 - singleslit_diffraction(x_slit-x_wire[j], aperature=wire, D=D) # diffraction pattern that passes slit from wire
        dx = x_slit[1] - x_slit[0]
        signal[j] = I[np.abs(x_slit)<slit/2].sum() * dx
    
    f = interp1d(x_wire, signal, bounds_error=True)
    return f



def signal_wireafterslit(slit=50, wire=50, D=1000):
    ''' Compute signal of photodiode with wire after slit.'''
    x_wire = np.linspace(-100,100,100000) # discretization at wire
    dx = x_wire[1]-x_wire[0]
    wire_window = round(wire/dx)
    I = singleslit_diffraction(x_wire, aperature=slit, D=D) #diffraction pattern at wire from slit
    blocked = np.convolve(I, dx*np.ones(wire_window), 'same') #integrate to get blocked signals for each x_wire
    signal = slit - blocked # measured at photodiode
    f = interp1d(x_wire, signal, bounds_error=True)
    return f

def plot_results(f, x, title, normed=True):
    signal = f(x)
    signal = signal*10/max(signal)
    derivative = np.append(0, np.diff(signal))
    
    if normed:
        derivative = derivative/derivative.max()

    # Plot
    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].set_title(title)
    axes[0].plot(x, signal)
    axes[0].set_ylabel('Signal (Volts)')
    axes[1].plot(x, derivative*1000)
    axes[1].set_ylabel('Derivative (mV/um)')
    axes[1].set_xlabel('Wire Position (um)')

    return



# ==== Parameters =============================================================
# All distance in um
# Assuming laser intensity is I0=1 (signal should have amplitude = slit)
D = 3000
wire = 50
x = np.linspace(-100,100,1000)
# =============================================================================
plt.close('all')
f50 = signal_wireafterslit(slit=50, wire=wire, D=D)
f100 = signal_wireafterslit(slit=100, wire=wire, D=D)
plot_results(f50, x, title='50um slit', normed=False)
plot_results(f100, x, title='100um slit', normed=False)


