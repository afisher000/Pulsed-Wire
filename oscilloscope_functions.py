# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:52:51 2022

@author: afisher
"""

# %%
import time
from pyvisa import ResourceManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Scope():
    def __init__(self, scope_id=None):
        if scope_id is None:
            scope_id = 'USB::0x0699::0x0412::C024123::INSTR'

        # Setup scope
        rm = ResourceManager()
        self.osc = rm.open_resource(scope_id)
        self.osc.timeout= 5000

        
    def get_measurements(self, channel, shots=10, npoints=100000, validate=True, 
        update_zero=False):
        # Set Acquisition settings
        self.osc.write(f'data:source ch{channel}')
        self.osc.write('data:start 1')
        self.osc.write(f'data:stop 100000')
        self.osc.write('ACQuire:STOPAfter SEQuence')
        
        shot_data_volts = np.zeros((shots, npoints))
        jshot = 0
        while jshot<shots:
            # Arm acquisition
            self.osc.write('ACQ:STATE ON')
            
            # Wait for acquisition
            while '1' in self.osc.query('BUSY?'):
                time.sleep(.1)
            
            # Query uint8 data
            shot_data_int8 = self.osc.query_binary_values( 
                'curve?', datatype='b', is_big_endian=True, container=np.array
            )%256 - 128

            # Convert to volts
            xinc = self.osc.query_ascii_values('wfmoutpre:xinc?')[0]
            xzero = self.osc.query_ascii_values('wfmoutpre:xzero?')[0]
            ymult = self.osc.query_ascii_values('wfmoutpre:ymult?')[0]
            yzero = self.osc.query_ascii_values('wfmoutpre:yzero?')[0]
            shot_data_volts[jshot, :] = yzero + ymult*shot_data_int8

            # Only move on to next shot if not input
            if validate:
                self.input = input('New Data:')
                print(self.input)
                if self.input=='':
                    jshot += 1
                elif self.input=='q':
                    jshot = shots #Break out of loop

            # Change offset to center on previous data
            if update_zero:
                new_offset = yzero + ymult*shot_data_int8[:100].mean()
                self.osc.write(f'ch{channel}:Offset {new_offset}')

        # Save to data and time attributes
        self.data = shot_data_volts
        self.time = xzero + xinc*np.arange(npoints)

        # Set scope back to continuous acquisition
        self.osc.write('ACQuire:STOPAfter Runstop')
        self.osc.write('ACQ:STATE ON')



    def save_measurements(self, filename):
        columns = [f'col{j}' for j in range(self.data.shape[0])]
        columns.insert(0, 'time')
        df = pd.DataFrame( 
            np.vstack([self.time, self.data]).T,
            columns = columns
        )

        # Create directory if necessary
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.mkdir(directory)
        df.to_csv(filename, index=False)
        return
            
            







# %%