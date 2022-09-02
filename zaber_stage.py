# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:35:14 2022

@author: afish
"""

# Have to install zaber_motion library using "pip install zaber_motion"
from zaber_motion import Library, Units
from zaber_motion.binary import Connection
import time


Library.enable_device_db_store()
with Connection.open_serial_port('COM3') as conn:
    # Connect and home stage
    stage = conn.detect_devices()[0]
    stage.home()
    time.sleep(1)
    
    # Take measurements
    for j in range(5):
        stage.move_relative(1, Units.LENGTH_MILLIMETRES)
        
    # Home at end of measurements
    stage.home()
    
    