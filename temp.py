# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:31:04 2023

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')
archive_folder = 'C:\\Users\\afisher\\Documents\\FASTGREENS DATA ARCHIVE\\THESEUS 2 PulsedWire Data'
folder = '2023-02-06 aligned without adjustment'
folder = '2023-02-06 Final Trajectory THES 2'
file = 'xtraj.csv'


path = os.path.join(archive_folder, folder, file)
df = pd.read_csv(path)
time = df.time
signal = df.drop(columns=['time']).mean(axis=1)

plt.plot(time, signal)
plt.show()