# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:15:00 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
plt.close('all')


shots = [1,2,3,4]
xamps = []
fig, ax = plt.subplots()
for shot in shots:
    file = f'shot{shot}.csv'
    data = pd.read_csv(file)
    data = data[data.time<2.5e-3]
    
    ax.plot(data.time, data.x-data.x.iloc[0])
    
    xdata = data[['time','x']].rename(columns={'x':'data'})
    amps = pwf.get_measurement_amplitudes(xdata, ref_magnet=False, annotate_plot=True)*1000
    print(f'Mean={amps.mean()}, Variation={amps.std()/amps.mean()}')
    xamps.append(amps.mean())
    