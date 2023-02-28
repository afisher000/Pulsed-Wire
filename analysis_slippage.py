# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:48:23 2023

@author: afisher
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_pulsedwire_edited as up
from scipy.signal import find_peaks, savgol_filter
plt.close('all')

archive_folder = 'C:\\Users\\afisher\\Documents\\Pulsed Wire Data Archive\\THESEUS 1 PulsedWire Data'
# archive_folder = ''

# wirescan_folder = '2022-12-19 xtraj, yoffset'
# wirescan_folder = '2022-12-21 ytraj, xoffset'
# wirescan_folder = '2022-12-21 xtraj, xoffset'
# wirescan_folder = '2022-12-28 xtraj, yoffset'
xtraj_wirescan_folder = '2022-12-28 xtraj, yoffset'
ytraj_wirescan_folder = '2022-12-28 ytraj, xoffset'

xtraj_file = os.path.join(archive_folder, xtraj_wirescan_folder, '(0,0).1.csv')
ytraj_file = os.path.join(archive_folder, ytraj_wirescan_folder, '(0,0).1.csv')



get_velocity_from_trajectory(time, signal, fix_dispersion=False, c0=207, EIwT=2*6.4e-8):
get_velocity_from_trajectory(time, signal, fix_dispersion=False, c0=207, EIwT=2*6.4e-8):



