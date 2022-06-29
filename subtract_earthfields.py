# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:45:04 2022

@author: afish
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\Pulse Wire Python Code\\CLEANED FILES\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
plt.close('all')

'''
##### WARNING #####
This code is unreliable until the non-symetric travelling waves on the string
can be understood.
'''

# Inputs
file = 'final_trajectory_shortened.csv'
current = 3.2


# Initialize series
# Units: [calibrations]=mV/um, [pulse current]=A, [amplitudes]=mV
calx, caly, cur, ampx, ampy = [pd.Series(dtype = 'float64') for _ in range (5)]
cur['earth'] = 3.2 * (3/10) #Not known for sure
cur['ref'] = 3.2 #Confirmed
cur['meas'] = current


# Read in calibrations
earth_cal = pd.read_csv(os.path.join('Earth Calibration', 'calibration_data.csv'))
ref_cal = pd.read_csv(os.path.join('Signal Calibration', 'calibration_data.csv'))

calx['earth'] = np.polyfit(earth_cal.dist, earth_cal.xvolts, 1)[0]*1000
caly['earth'] = np.polyfit(earth_cal.dist, earth_cal.yvolts, 1)[0]*1000
calx['ref'] = np.polyfit(ref_cal.dist, ref_cal.xvolts, 1)[0]*1000
caly['ref'] = np.polyfit(ref_cal.dist, ref_cal.yvolts, 1)[0]*1000
calx = calx.abs() # Calibrations should be positive
caly = caly.abs()


# Read in mean amplitudes
ref_signal = pd.read_csv(os.path.join('Signal Calibration', 'calibration_signal.csv'))
xref = ref_signal[['time','x']].rename(columns={'x':'data'})
yref = ref_signal[['time','y']].rename(columns={'y':'data'})
ampx['ref'] = pwf.get_measurement_amplitudes(xref, ref_magnet=False).mean()*1000
ampy['ref'] = pwf.get_measurement_amplitudes(yref, ref_magnet=False).mean()*1000


# Find mean amplitudes of measured signal
meas_signal = pd.read_csv(file)
xmeas = meas_signal[['time','x']].rename(columns={'x':'data'})
ymeas = meas_signal[['time','y']].rename(columns={'y':'data'})
ampx['meas'] = pwf.get_measurement_amplitudes(xmeas, ref_magnet=False).mean()*1000
ampy['meas'] = pwf.get_measurement_amplitudes(ymeas, ref_magnet=False).mean()*1000


# Compute correction from earth signal
earth_signal = pd.read_csv(os.path.join('Earth Calibration', 'calibration_signal.csv'))
earth_signal.x = earth_signal.x - earth_signal.x.iloc[0]
earth_signal.y = earth_signal.y - earth_signal.y.iloc[0]
earth_xinterp = interp1d(earth_signal.time, savgol_filter(earth_signal.x, 151, 3))
earth_yinterp = interp1d(earth_signal.time, savgol_filter(earth_signal.y, 151, 3))

# Check these equations again...
x_correction = earth_xinterp(meas_signal.time) / (cur.earth*calx.earth) * (cur.ref*calx.ref) * (ampx.meas/ampx.ref)
y_correction = earth_yinterp(meas_signal.time) / (cur.earth*caly.earth) * (cur.ref*caly.ref) * (ampy.meas/ampy.ref)


# Apply correction
meas_signal['x_correct'] = meas_signal.x - x_correction
meas_signal['y_correct'] = meas_signal.y - y_correction


# Plot
ax = meas_signal.plot(x='time', y=['x','x_correct'])
#ax.plot(df.time, np.ones_like(df.time)*df.x[0], c='k')
ax = meas_signal.plot(x='time', y=['y', 'y_correct'])
#ax.plot(df.time, np.ones_like(df.time)*df.y[0], c='k')









    


