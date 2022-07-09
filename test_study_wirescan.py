# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:32:55 2022

@author: afisher
"""
import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.close('all')
# path = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\April 2022 Pulse Wire\\Centering Wire'
# file = '2022-05-12 (ytraj, xoffset) Andrew.csv'
#file = '2022-06-01 (ytraj, xoffset) Andrew.csv'


path = ''
file = '2022-07-08 (xtraj, xoffset).csv'

# Parse trajectory and offset direction
traj = file[file.find('traj')-1]
offset = file[file.find('offset')-1] + 'offset'
    
df = pd.read_csv(os.path.join(path, file)).set_index(['iteration',offset])
df = df.sort_index(level=0)
df = df.loc[1] #Only consider first iteration
df = df.iloc[df.index<=750]




# Make dataframe for means and amplitudes

fig, ax = plt.subplots()
data = pd.DataFrame(columns=['amps','means','offset'])
for meas_offset in df.index.unique():
    meas = df.loc[meas_offset].copy()
    meas.rename(columns = {traj:'data'}, inplace=True)
    amplitudes, means = pwf.get_measurement_amplitudes(meas, 
                                                       ref_magnet=False,
                                                       return_means = True)
    
    temp_df = pd.DataFrame(np.array([amplitudes, 
                                     means, 
                                     meas_offset*np.ones_like(means)
                                     ]).T, 
                           columns=['amps','means','offset'])
    data = pd.concat([data, temp_df])
    ax.plot(meas.time, meas.data.values-df.loc[0].x.values+df.loc[0].x.values[0]-meas.data.values[0], label=meas_offset)
ax.legend()


cmap = plt.get_cmap('coolwarm')
data.plot.scatter(x='means', y='amps', c='offset', cmap=cmap) 
data.plot.scatter(x='offset', y='amps', c = data.index, cmap=cmap)
ax = data.plot.scatter(x='offset', y='amps')
qfit = np.polyfit(data.offset, data.amps, 2)
offsets = np.sort(data.offset.unique())
ax.plot(offsets, np.polyval(qfit, offsets))
axis, extrema = pwf.fit_concavity(data.offset, data.amps)

data['errors'] = data.amps - np.polyval(qfit, data.offset)
data.plot.scatter(x='means', y='errors', c='offset', cmap = plt.get_cmap('coolwarm'))



# =============================================================================
# def scale_amplitude(data, x0):
#     poly = np.polyfit(data.means, data.amps, 2)
#     amp0 = np.polyval(poly, x0)
#     return data.amps/amp0
#
# =============================================================================

    

    