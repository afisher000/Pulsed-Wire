# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:50:51 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
plt.close('all')


# pwf.analyze_wirescan('2022-07-12 (xtraj, yoffset).csv')
def plot_theory(ax, wire_position, sign=1, label='Theory'):
    theory_offsets = np.linspace(-1500,1500,100)
    ku = 2*np.pi/32e3 #um
    if sign>0:
        coeff = 1.53
        sign = 1
    else:
        coeff = 1.16
        sign = -1

    theory_amplitudes = 1 + sign*0.5*(ku*coeff*(theory_offsets-wire_position))**2
    ax.plot(theory_offsets, theory_amplitudes, label=label)
    return


# path = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\April 2022 Pulse Wire\\Centering Wire'
path = ''
xyfile = '2022-07-12 (xtraj, yoffset).csv'
yyfile = '2022-07-12 (ytraj, yoffset).csv'
xxfile = '2022-07-12 (xtraj, xoffset).csv'
yxfile = '2022-07-12 (ytraj, xoffset).csv'

xydata, xypeak_data = pwf.analyze_wirescan(xyfile, plot=False, remove_dispersion=False)
yydata, yypeak_data = pwf.analyze_wirescan(yyfile, plot=False, remove_dispersion=False)
xxdata, xxpeak_data = pwf.analyze_wirescan(xxfile, plot=False, remove_dispersion=False)
yxdata, yxpeak_data = pwf.analyze_wirescan(yxfile, plot=False, remove_dispersion=False)



cmap = plt.get_cmap('coolwarm')

# Plot single peak concavity plot
pk = 0
fig, ax = plt.subplots()
ax.scatter(xydata.loc[pk].offset, xydata.loc[pk].amps/xypeak_data.extrema.loc[pk],
           label='Y traj.')
plot_theory(ax, xypeak_data.axis.loc[pk], sign=1, label='Thy: alpha=1.53')
ax.scatter(xxdata.loc[pk].offset, xxdata.loc[pk].amps/xxpeak_data.extrema.loc[pk],
           label='X traj.')
plot_theory(ax, xxpeak_data.axis.loc[pk], sign=-1, label='Thy: beta=1.16')
ax.set_xlabel('Wire Displacement in Y (um)')
ax.set_ylabel('Norm. Amplitude')
ax.legend()

# Plot single peak concavity plot
fig, ax = plt.subplots()
ax.scatter(yxdata.loc[0].offset, yxdata.loc[0].amps/yxpeak_data.extrema.loc[0])
plot_theory(ax, yxpeak_data.axis.loc[0], sign=1)
ax.scatter(xxdata.loc[0].offset, xxdata.loc[0].amps/xxpeak_data.extrema.loc[0])
plot_theory(ax, xxpeak_data.axis.loc[0], sign=-1)
ax.set_xlabel('Wire Displacement in X (um)')
ax.set_ylabel('Norm. Peak Amplitude')

# Plot concavity plots
fig, ax = plt.subplots()
ax.scatter(xydata.offset, xydata.amps/xypeak_data.extrema.mean(), 
           label='Xtraj data', c=xydata.index)
plot_theory(ax, xypeak_data.axis.mean(), 1)
ax.scatter(yydata.offset, yydata.amps/yypeak_data.extrema.mean(), label='Ytraj data',
           c=yydata.index)
plot_theory(ax, yypeak_data.axis.mean(), -1)
ax.legend()
ax.set_xlabel('Yoffset')
ax.set_ylabel('Normalized Amplitude')
fig.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)

# Plot concavity plots
fig, ax = plt.subplots()
ax.scatter(xxdata.offset, xxdata.amps/xxpeak_data.extrema.mean(), 
           label='Xtraj data', c=xxdata.index)
plot_theory(ax, xxpeak_data.axis.mean(), 1)
ax.scatter(yxdata.offset, yxdata.amps/yxpeak_data.extrema.mean(), label='Ytraj data',
           c=yxdata.index)
plot_theory(ax, yxpeak_data.axis.mean(), -1)
ax.legend()
ax.set_xlabel('Xoffset')
ax.set_ylabel('Normalized Amplitude')
fig.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)


# Plot wire alignment
fig, ax = plt.subplots()
ax.plot(xypeak_data.peak, xypeak_data.axis, label='Xtraj data')
ax.plot(yypeak_data.peak, yypeak_data.axis, label='Ytraj data')
ax.legend()
ax.set_ylabel('Yoffset')
ax.set_xlabel('Peak Number')

fig, ax = plt.subplots()
ax.plot(xxpeak_data.peak, xxpeak_data.axis, label='Xtraj data')
ax.plot(yxpeak_data.peak, yxpeak_data.axis, label='Ytraj data')
ax.legend()
ax.set_ylabel('Xoffset')
ax.set_xlabel('Peak Number')
