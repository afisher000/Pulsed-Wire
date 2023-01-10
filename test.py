# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:26:55 2022

@author: afisher
"""
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_pulsedwire_edited import get_signal_means_and_amplitudes
from utils_oscilloscope import Scope
import seaborn as sns
plt.close('all')

def get_field_areas(m, orig_field):
    field = orig_field[(m>20)&(m<90)]
    x = np.cumsum(field)
    zc = np.where(np.sign(field[:-1])!=np.sign(field[1:]))[0]
    areas = np.diff(x[zc])
    return areas

data0 = pd.read_csv('hall_probe_fields_scan0.csv') # Rescan, no changes, before moving T80, straight traj
data1 = pd.read_csv('hall_probe_fields_scan1.csv') # Make 5 adjust, before moving T80, straight traj
data2 = pd.read_csv('hall_probe_fields_kick_at_80.csv') # After moving T80, kick down
data3 = pd.read_csv('hall_probe_fields_kick_at_77.csv') # After moving T80, kick up

# Computations
m = data0.m.values
datas = [data0, data1, data2, data3]
fields = [data.By.values for data in datas]
Bsquareds = [data.Bx.values**2 + data.By.values**2 for data in datas]

areas = np.array([get_field_areas(m, field) for field in fields])
norm_areas = areas/np.abs(areas).mean(axis=0)
shifted_areas = norm_areas - np.sign(norm_areas)




# # Field comparison
# fig, ax = plt.subplots()
# ax.plot(m, fields[0], label='0')
# ax.plot(m, fields[1], label='1')
# ax.plot(m, fields[2], label='2')
# ax.plot(m, fields[3], label='3')
# ax.legend()
# ax.set_xlim([70, 85])
# ax.set_ylabel('fields')
# ax.set_xlabel('Magnet number')

# # BSquared
# fig, ax = plt.subplots()
# ax.plot(m, Bsquareds[2], label='2')
# ax.plot(m, Bsquareds[3], label='3')
# ax.legend()
# ax.set_xlim([70, 85])
# ax.set_ylabel('Bx^2 + By^2')
# ax.set_xlabel('Magnet Number')


# # Areas
fig, ax = plt.subplots()
m_areas = np.arange(22, 90, 2)
ax.scatter(m_areas, shifted_areas[0], label='0')
ax.scatter(m_areas, shifted_areas[1], label='1')
# ax.scatter(m_areas, shifted_areas[2], label='2')
# ax.scatter(m_areas, shifted_areas[3], label='3')
# ax.scatter(m_areas, shifted_areas[1]-shifted_areas[0])
# ax.scatter(m_areas, shifted_areas[3]-shifted_areas[2])
ax.legend()
ax.set_xlabel('Magnet Number')
ax.set_ylabel('Area between zeros')




