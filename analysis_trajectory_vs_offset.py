# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:12:08 2023

@author: afisher
"""
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')

# archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive\\THESEUS 1 PulsedWire Data'
archive_folder = ''


# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-22 xtraj, yoffset'
# wirescan_folder = '2022-12-28 xtraj, yoffset'
# wirescan_folder = '2022-12-28 ytraj, xoffset'
# wirescan_folder = '2023-01-20 xtraj, yoffset'
# wirescan_folder = '2023-01-23 ytraj, xoffset'


# wirescan_folder = '2023-05-31 xtraj, yoffset'
wirescan_folder = '2023-06-01 ytraj, xoffset straightness3'

folder = os.path.join(archive_folder, wirescan_folder)

# Get reference
file = '(0,0).csv'
df = pd.read_csv(os.path.join(folder, file))
time = df.time.values
ref_signal = df.drop(columns=['time']).mean(axis=1)
ref_signal = ref_signal - ref_signal[4000]
# ref_signal = 0


# Loop over offsets
fig, ax = plt.subplots()
for file in os.listdir(folder):
    if file=='calibration.csv':
        continue

    #if file != '(-1000,0).csv':
    #    continue
    
    leftpar = file.find('(')
    comma = file.find(',')
    rightpar = file.find(')')
    
    if folder.find('xoffset')!=-1:
        offset = file[leftpar+1:comma]
    else:
        offset = file[comma+1:rightpar]
    # print(offset)
    df = pd.read_csv(os.path.join(folder, file))
    
    time = df.time.values
    signal = df.drop(columns=['time']).mean(axis=1)
    # signal = signal - signal[0]
    signal = signal - ref_signal
    signal = signal - signal[4000]
    ax.plot(time, signal, label=offset)
ax.legend()
# %%
