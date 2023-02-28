# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:12:08 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')

archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive\\THESEUS 1 PulsedWire Data'

# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-22 xtraj, yoffset'
# wirescan_folder = '2022-12-28 xtraj, yoffset'
# wirescan_folder = '2022-12-28 ytraj, xoffset'
# wirescan_folder = '2023-01-20 xtraj, yoffset'
wirescan_folder = '2023-01-23 ytraj, xoffset'

folder = os.path.join(archive_folder, wirescan_folder)

fig, ax = plt.subplots()
for file in os.listdir(folder):
    if file=='calibration.csv':
        continue
    
    leftpar = file.find('(')
    comma = file.find(',')
    rightpar = file.find(')')
    
    if folder.endswith('xoffset'):
        offset = file[leftpar+1:comma]
    else:
        offset = file[comma+1:rightpar]
    # print(offset)
    df = pd.read_csv(os.path.join(folder, file))
    
    time = df.time.values
    signal = df.drop(columns=['time']).mean(axis=1)
    signal = signal - signal[0]
    ax.plot(time, signal, label=offset)

ax.legend()