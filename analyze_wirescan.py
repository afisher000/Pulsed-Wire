# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:50:51 2022

@author: afisher
"""

import sys
sys.path.append('C:\\Users\\afisher\\Documents\\GitHub\\Pulsed-Wire\\PythonPackages')
import pulsedwire as pwf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.close('all')




# path = 'C:\\Users\\afisher\\Documents\\Magnet Tuning\\Summer 2022 FASTGREENS\\April 2022 Pulse Wire\\Centering Wire'
path = ''
xyfile = '2022-07-11 (xtraj, yoffset).csv'
yyfile = '2022-07-11 (ytraj, yoffset).csv'
xxfile = '2022-07-11 (xtraj, xoffset).csv'
yxfile = '2022-07-11 (ytraj, xoffset).csv'

xydata, xypeak_data = pwf.analyze_wirescan(xyfile, plot=False)
yydata, yypeak_data = pwf.analyze_wirescan(yyfile, plot=False)
xxdata, xxpeak_data = pwf.analyze_wirescan(xxfile, plot=False)
yxdata, yxpeak_data = pwf.analyze_wirescan(yxfile, plot=False)


# Plot concavity plots
fig, ax = plt.subplots()
ax.scatter(xydata.offset, xydata.amps/xydata.amps.min(), label='Xtraj data')
ax.scatter(yydata.offset, yydata.amps/yydata.amps.max(), label='Ytraj data')
ax.legend()
ax.set_xlabel('Yoffset')
ax.set_ylabel('Normalized Amplitude')

# Plot concavity plots
fig, ax = plt.subplots()
ax.scatter(xxdata.offset, xxdata.amps/xxdata.amps.max(), label='Xtraj data')
ax.scatter(yxdata.offset, yxdata.amps/yxdata.amps.min(), label='Ytraj data')
ax.legend()
ax.set_xlabel('Xoffset')
ax.set_ylabel('Normalized Amplitude')

# =============================================================================
# # Plot wire alignment
# fig, ax = plt.subplots()
# ax.plot(xypeak_data.peak, xypeak_data.axis, label='Xtraj data')
# ax.plot(yypeak_data.peak, yypeak_data.axis, label='Ytraj data')
# ax.legend()
# ax.set_ylabel('Yoffset')
# ax.set_xlabel('Peak Number')
# 
# fig, ax = plt.subplots()
# ax.plot(xxpeak_data.peak, xxpeak_data.axis, label='Xtraj data')
# ax.plot(yxpeak_data.peak, yxpeak_data.axis, label='Ytraj data')
# ax.legend()
# ax.set_ylabel('Xoffset')
# ax.set_xlabel('Peak Number')
# =============================================================================
