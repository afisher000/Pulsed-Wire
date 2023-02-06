# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:35:41 2022

@author: afisher
"""



import pandas as pd
import numpy as np
import os
import re

save_to_name = 'laserscan y.csv'
folder = os.path.join('Ysignal Scan')
files = [file for file in os.listdir(folder) if 'signal' in file]

dfs = []
for file in files:
    df = pd.read_csv(os.path.join(folder,file))
    offset = re.search('(-?\d*)um', file).groups()[0]
    df['dist'] = int(offset)
    df.set_index('dist', drop=True, inplace=True)
    dfs.append(df)
        
df = pd.concat(dfs)
df.to_csv(os.path.join(folder, save_to_name))