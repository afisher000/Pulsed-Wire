# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:51:53 2022

@author: afisher
"""


import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
import os
import re
plt.close('all')

folder = '2022-07-13 ytraj calibration'
disp = []
for file in os.listdir(folder):
    regex = re.compile('(-?[\d]*)um.csv')
    result = regex.search(file)
    if result:
        disp.append(int(result.groups()[0]))
        
        
        
# Check signal amplitudes vs disp
amps = np.zeros(len(disp))
zeros = np.zeros(len(disp))
for j in range(len(disp)):
    file = os.path.join(folder,f'{disp[j]}um.csv')
    meas = pd.read_csv(file)
    meas.columns = ['data','time']
    amp = pwf.get_measurement_amplitudes(meas, ref_magnet=False).mean()*1000 #mV
    zeros[j] = meas.data.iloc[0]
    amps[j] = amp
    print(f'{file}: amp={amp:.1f}')

data = pd.DataFrame(np.array([disp, zeros, amps]).T, columns=['um','zeros','amps'])

# Linear region
ax_linearfit = data.plot.scatter(x='um', y='zeros')
trun_data = data[data.amps>data.amps.max()*.8]
poly = np.polyfit(trun_data.um, trun_data.zeros, 1)
ax_linearfit.plot(trun_data.um, np.polyval(poly, trun_data.um), label=f'Fit={abs(poly[0]*1000):.1f}')
ax_linearfit.legend()

# Signal amplitudes vs um
data.plot.scatter(x='um', y='amps')

# Signal amplitudes vs zeros
data.plot.scatter(x='zeros', y='amps')
