# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:39:45 2022

@author: afish
"""
# %%
import sys
sys.path.append('C:\\Users\\afish\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
from oscilloscope import Scope
import pulsedwire as pwf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

folder = '2022-09-02 (X Calibration)'
file = 'file0.csv'
filename = os.path.join(folder, file)
coord = 'x'
channel = 2

scope = Scope()
scope.get_measurements(channel=channel, shots=4, validate=True)
scope.save_measurements(filename=filename, coord=coord)

pickle.dump(scope.data, open('data.pkl', 'wb'))
pickle.dump(scope.time, open('time.pkl', 'wb'))










    





# %%
