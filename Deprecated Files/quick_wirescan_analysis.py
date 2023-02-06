# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 06:40:47 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import oscilloscope as osc
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.close('all')
folder = '2022-09-02 (ytraj, xoffset)'

pwf.merge_wirescan_files(folder)
data, pkdata = pwf.analyze_wirescan(file = folder+'.csv', plot=False)

fig, ax = plt.subplots()
pkdata.plot.scatter(x='peak',y='axis', ax=ax)
fit = np.polyfit(pkdata.peak, pkdata.axis, 1)
ax.plot(pkdata.peak, np.polyval(fit, pkdata.peak), c='k')



# Estimate wire adjustments
dist_tuple = np.array([45, 39, 5])*.0254 # dist between p1, und endplates, and p2
dz = (dist_tuple[1]-(len(pkdata)-1)*0.032/2)/2 
pkdata['z'] = pkdata.peak*.032/2 + dz + dist_tuple[0]

fit = np.polyfit(pkdata.z, pkdata.axis, 1)
p1_adjust = np.polyval(fit, 0)
p2_adjust = np.polyval(fit, sum(dist_tuple))
print(f'P1_adjust = {p1_adjust:.0f}\nP2_adjust = {p2_adjust:.0f}')


# Concavity plots
fig, ax = plt.subplots()
ax.scatter(data.offset, data.amps/pkdata.extrema.mean(), 
           label='Xtraj data', c=data.index)


ax.legend()
ax.set_xlabel('Yoffset')
ax.set_ylabel('Normalized Amplitude')
fig.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)

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
