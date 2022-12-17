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
import visa
import time
import pyvisa


scope_id = 'USB0::0x699::0x408::C031986::INSTR'

#Connect to the instrument
rm = pyvisa.ResourceManager()
scope = rm.open_resource(scope_id)


#Start single sequence acquisition
scope.write("ACQ:STOPA SEQ")
loop = 0

while True:
    #increment the loop counter
    loop += 1
    print (f'On Loop {loop}')
    
    #Arm trigger, then loop until scope has triggered
    scope.write("ACQ:STATE ON")
    while '1' in scope.ask("ACQ:STATE?"):
        time.sleep(0.5)

    #save all waveforms, then wait for the waveforms to be written
    scope.write("SAVE:WAVEFORM ALL, \"E:/Saves/All_%s\"" %loop)
    while '1' in scope.ask("BUSY?"):
        time.sleep(0.5)
