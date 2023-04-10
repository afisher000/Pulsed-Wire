# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:31:04 2023

@author: afisher
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')


folder = 'C:\\Users\\afish\\Documents\\Fermilab Measurements\\Theseus 2'

file = 'UCLA xtraj.csv'
dfx = pd.read_csv(os.path.join(folder, file))

file = 'UCLA ytraj.csv'
dfy = pd.read_csv(os.path.join(folder, file))

time = dfx.time.values
xtraj = dfx.drop(columns=['time']).mean(axis=1)
ytraj = dfy.drop(columns=['time']).mean(axis=1)

final_traj = pd.DataFrame(
    data = np.vstack([time, xtraj, ytraj]).T,
    columns=['time','xtraj','ytraj']
)
final_traj.to_csv(os.path.join(folder, 'UCLA final trajectory waveforms.csv'), index=False)
# %%
