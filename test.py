# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:19:29 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('2022-09-02 (xtraj, yoffset).csv')

pos0 = df[df.yoffset==0]
neg500= df[df.yoffset==500]
pos500 = df[df.yoffset==-500]

fig, ax = plt.subplots()
pos0.plot('time', 'x', ax=ax)
neg500.plot('time','x', ax=ax)
pos500.plot('time', 'x', ax=ax)

df = pd.read_csv('2022-09-02 (ytraj, xoffset).csv')
pos0 = df[df.xoffset==0]
neg500= df[df.xoffset==500]
pos500 = df[df.xoffset==-500]

fig, ax = plt.subplots()
pos0.plot('time', 'y', ax=ax)
neg500.plot('time','y', ax=ax)
pos500.plot('time', 'y', ax=ax)