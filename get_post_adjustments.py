# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:13:24 2022

@author: afish

Script to compute adjustments for posts in pulsed wire setup to move wire
on axis given offsets at peak locations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pk_offsets = np.array([203, 231, 294, 284, 245, 369, 243, 265, 371, 383, 534, 355, 815, 1191, 1049])
pk_idx = np.arange(15)




