# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:52:51 2022

@author: afisher
"""

#-------------------------------------------------------------------------------
#  Save All waveforms to USB every time scope triggers

# python        2.7         (http://www.python.org/)
# pyvisa        1.4         (http://pyvisa.sourceforge.net/)
#-------------------------------------------------------------------------------

# %%
import time
from pyvisa import ResourceManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Scope():
    def __init__(self, scope_id=None):
        if scope_id is None:
            scope_id = 'USB::0x0699::0x0412::C024123::INSTR'

        # Setup scope
        rm = ResourceManager()
        self.osc = rm.open_resource(scope_id)
        self.osc.timeout= 5000
        
        # Read waveform settings
        self.udpate_wfm_settings()

    def update_wfm_settings(self):
        ''' Return a pandas Series object containing waveform (wfm) settings '''
        values = np.array([
            self.osc.query_ascii_values('wfmoutpre:xinc?')[0],
            self.osc.query_ascii_values('wfmoutpre:xzero?')[0],
            self.osc.query_ascii_values('wfmoutpre:ymult?')[0],
            self.osc.query_ascii_values('wfmoutpre:yoff?')[0],
            self.osc.query_ascii_values('wfmoutpre:yzero?')[0],
            self.osc.query_ascii_values('wfmoutpre:nr_pt?')[0]
            ])
    
        names = ['xinc','xzero','ymult','yoff','yzero','points']
        self.wfm = pd.Series(values, index=names)
        return
        
    def get_measurements(self, channel, shots=10, npoints=100000):
        # Set Acquisition settings
        self.osc.write(f'data:source ch{channel}')
        self.osc.write('data:start 1')
        self.osc.write(f'data:stop 100000')
        self.osc.write('ACQuire:STOPAfter SEQuence')
        
        for j in range(shots):
            # Arm acquisition
            self.osc.write('ACQ:STATE ON')
            
            # Wait for acquisition
            while '1' in self.osc.query('BUSY?'):
                time.sleep(.1)
            
            # Query uint8 data (might need modulus 256)
            uint8_data = self.osc.query_binary_values( 
                'curve?', datatype='b', is_big_endian=True, container=np.array
            )
            
            
            
            
            
            

