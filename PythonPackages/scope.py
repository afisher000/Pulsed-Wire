# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:47:03 2022

@author: afish
"""

import pyvisa
import numpy as np
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import os


def get_wfm_settings(scope):
    # Return a pandas Series object containing waveform settings
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

def setup_scope(scope_id):
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(scope_id)
    scope.timeout= 8000
    scope.write('data:source ch1')
    scope.write('data:start 1')
    scope.write('data:stop 100000')
    return scope

def request_measurement(scope, channel_map, trigger_check=0):
    '''Reads all channels requested and saves in dict named data indexed by 
    axis (x or y). If mean of first measurement matches prev_meas, return None. '''
    
    data={}
    for channel,axis in channel_map.items():
        scope.write(f'data:source {channel}')
        data[axis] = scope.query_binary_values('curve?', 
                        datatype='b', is_big_endian=True, container=np.array)
        
        # Return none if duplicate measurement
        if data[axis].sum()==trigger_check:
            return None
    return pd.DataFrame(data)

def read_measurements(scope_id, max_meas=20, channel_map={'ch1':'x'}, rep_rate=1.3,
                      filename=None, average_only=True):
    '''Build list of dataframes containing data. '''
    scope = setup_scope(scope_id)
    
    j = 0
    data_list = []
    new_data = request_measurement(scope, channel_map)
    trigger_check = new_data.values.sum(axis=0)[0]
    data_list.append(new_data)
    while j+1<max_meas:
        new_data = request_measurement(scope, channel_map, trigger_check=trigger_check)
        if new_data is not None:
            print(f'Measurement {j+1}')
            data_list.append(new_data)
            trigger_check = new_data.values.sum(axis=0)[0]
            j+=1
        else:
            print('Read same data')
        sleep(1/rep_rate)
    
    # Average Measurements
    df = pd.DataFrame(np.array(data_list).sum(axis=0)/max_meas,
                      columns=channel_map.values()
                      )
    
    # Scale values
    for channel, axis in channel_map.items():
        wfm = get_wfm_settings(scope)
        df[axis] = wfm.yzero + wfm.ymult*df[axis]
    df['time'] = wfm.xzero + wfm.xinc*np.arange(wfm.points)
    if filename is not None:
        if os.path.exists(filename):
            print(f'File already exists!!!')
        else:
            df.to_csv(filename, index=False)
            print(f'Data saved to {filename}.')
    return df

def remove_noisy_shots(dirty_df, thresh_fac=3, noise_quantile=.9, plot=False):
    # Subtract average trajectory and slow variations
    dirty_df.set_index('time', inplace=True)
    dirty_average = dirty_df.mean(axis=1)
    deviation_from_average = dirty_df.apply(lambda x: x-dirty_average)
    
    def remove_slow_variation(series):
        vec = np.arange(len(series))
        poly = np.polyfit(vec, series, 3)
        return series - np.polyval(poly, vec)
    
    noise = deviation_from_average.apply(remove_slow_variation)
    noise_thresh = thresh_fac*np.quantile(np.abs(noise.values.ravel()),
                                          noise_quantile)
    
    bad_columns = noise.columns[(noise>noise_thresh).any()]
    print(f'Dropped {len(bad_columns)} noisy measurements out of {len(noise.columns)} total.')
    clean_df = dirty_df.drop(columns = bad_columns)
    clean_df['volts'] = clean_df.mean(axis=1)
    
    if plot:
        fig, ax = plt.subplots()
        [ax.plot(noise.index, noise[str(j)]) for j in np.arange(20)]
        
        fig, ax = plt.subplots()
        ax.plot(dirty_df.index, dirty_average)
        ax.plot(clean_df.index, clean_df.volts)
        
        
    clean_df.reset_index(inplace=True)
    return clean_df
