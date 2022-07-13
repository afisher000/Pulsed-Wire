# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:26:42 2022

@author: afisher
"""

import numpy as np
import pandas as pd

radius = 25e-6 #m
density = 8.25/1000*1e6 #kg/m^3
tension = 1 #N
EIw = 2e-7 #N*m^2 (from paper)
linear_density = density * (np.pi*radius**2)



