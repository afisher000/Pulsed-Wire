# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:51:53 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import numpy as np
import pandas as pd
import pulsedwire as pwf
import matplotlib.pyplot as plt
import os
plt.close('all')

# Check signal amplitudes vs disp
laserscan1 = pd.read_csv('laserscan1.csv').set_index('dist', drop=True)
laserscan2 = pd.read_csv('laserscan2.csv').set_index('dist', drop=True)

deflection1 = pwf.analyze_laserscan(laserscan1, 'y', title='YTraj Laser - Wire farther from slit')
deflection2 = pwf.analyze_laserscan(laserscan2, 'y', title='YTraj Laser - Wire closer to slit')



