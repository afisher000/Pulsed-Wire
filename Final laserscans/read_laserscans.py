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

# NOTES: Xlaserscan was done at 1.95 Amps and Ylaserscan was done at 0.76 Amps.

# Check signal amplitudes vs disp
laserscanx = pd.read_csv('laserscan x.csv').set_index('dist', drop=True)
laserscany = pd.read_csv('laserscan y.csv').set_index('dist', drop=True)

xdeflection = pwf.analyze_laserscan(laserscanx, 'x', title='XTraj Laser (1.95 Amps)')
ydeflection = pwf.analyze_laserscan(laserscany, 'y', title='YTraj Laser (0.76 Amps)')
