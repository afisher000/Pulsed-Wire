# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:47:03 2022

@author: afish
"""

import pyvisa
import numpy as np
import pandas as pd
from time import sleep
import os


def get_wfm_settings(scope):
    ''' Return a pandas Series object containing waveform (wfm) settings '''
    values = np.array([
        scope.query_ascii_values('wfmoutpre:xinc?')[0],
        scope.query_ascii_values('wfmoutpre:xzero?')[0],
        scope.query_ascii_values('wfmoutpre:ymult?')[0],
        scope.query_ascii_values('wfmoutpre:yoff?')[0],
        scope.query_ascii_values('wfmoutpre:yzero?')[0],
        scope.query_ascii_values('wfmoutpre:nr_pt?')[0]
        ])

    names = ['xinc','xzero','ymult','yoff','yzero','points']
    wfm = pd.Series(values, index=names)
    return wfm

def setup_scope(scope_id, npoints=100000):
    '''Setup the scope. Set to read npoints at a time.'''
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(scope_id)
    scope.timeout= 5000
    scope.write('data:source ch1')
    scope.write('data:start 1')
    scope.write(f'data:stop {npoints}')
    return scope

def check_measurement(scope, channel_map, trigger_check=0):
    '''Reads all channels requested and returns in dataframe with columns named
    by coordinate. If the mean of the first measurement matches trigger_check,
    no trigger occured between readings and None is returned.'''
    
    data={}
    for channel,axis in channel_map.items():
        scope.write(f'data:source {channel}')
        data[axis] = scope.query_binary_values('curve?', 
                        datatype='b', is_big_endian=True, container=np.array)
        
        # Return none if duplicate measurement
        if data[axis].sum()==trigger_check:
            return None
    return pd.DataFrame(data)

def get_measurements(scope, max_meas=20, channel_map={'ch1':'x'}, rep_rate=1.3,
                      filename=None, average_only=True):
    '''Build list of dataframes containing data. When reading measurements,
    check that data changed before incrementing (do not think there is another  
    way to check if trigger occured). Compute the average, scale according to 
    the waveform settings, and save to file is filename is not None.'''
    
    j = 0
    data_list = []
    new_data = check_measurement(scope, channel_map)
    data_list.append(new_data)
    trigger_check = new_data.values.sum(axis=0)[0]
    
    while j+1<max_meas:
        sleep(1/rep_rate)
        new_data = check_measurement(scope, channel_map, trigger_check=trigger_check)
        if new_data is not None:
            print(f'Measurement {j+1}')
            data_list.append(new_data)
            trigger_check = new_data.values.sum(axis=0)[0]
            j+=1
        else:
            print('Read same data')
        
    # Average Measurements
    df = pd.DataFrame(np.array(data_list).sum(axis=0)/max_meas,
                      columns=channel_map.values())
    
    # Scale uint8 data with waveform settings
    wfm = get_wfm_settings(scope)
    for channel, axis in channel_map.items():
        df[axis] = wfm.yzero + wfm.ymult*df[axis]
    df['time'] = wfm.xzero + wfm.xinc*np.arange(wfm.points)
    
    # Save to file
    if filename is not None:
        if os.path.exists(filename):
            print('File already exists!!!')
        else:
            df.to_csv(filename, index=False)
            print(f'Data saved to {filename}.')
    else:
        print('File not saved.')
    return df

