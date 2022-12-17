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
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


scope_id = 'USB::0x0699::0x0412::C024123::INSTR'
event_type = pyvisa.constants.EventType.trig

#Connect to the instrument
rm = pyvisa.ResourceManager()
scope = rm.open_resource(scope_id)
scope.timeout= 5000

# Setup trigger
scope.write('TRIGger:EDGe:SOUrce CH1')
scope.write('TRIGger:EDGe:SLOpe RISing')
scope.write('ACQuire:STOPAfter SEQuence')

for j in range(3):
    # Set to acquire
    scope.write('ACQ:STATE ON')
    while scope.query('BUSY?'):
        print('Scope is busy..')
        time.sleep(.1)
    while 'SAV' not in scope.query('TRIGger:State?'):
        print('Not triggered, wait..')
        time.sleep(.5)
    print(j)
    data = scope.query_binary_values('curve?', 
                    datatype='b', is_big_endian=True, container=np.array)%256


# scope.write('TRIGger')
# scope.query('TRIGger:STATus?')

# # Plot waveform
# for j in range(5):
#     test = scope.query_binary_values('curve?', 
#                     datatype='b', is_big_endian=True, container=np.array)%256

#     plt.plot(test)
#     plt.show()



# loop = 0

# while True:
#     #increment the loop counter
#     loop += 1
#     print (f'On Loop {loop}')
    
#     #Arm trigger, then loop until scope has triggered
#     scope.write("ACQ:STATE ON")
#     while '1' in scope.ask("ACQ:STATE?"):
#         time.sleep(0.5)

#     #save all waveforms, then wait for the waveforms to be written
#     scope.write("SAVE:WAVEFORM ALL, \"E:/Saves/All_%s\"" %loop)
#     while '1' in scope.ask("BUSY?"):
#         time.sleep(0.5)

# %%
