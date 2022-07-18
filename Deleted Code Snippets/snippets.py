# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:04:46 2022

@author: afisher
"""


''' Code used to create linear_ramps in pulsedwire.get_fourier_transform()'''
# Edge fits
dt = td.time.diff().mean()
Npoints = round((.032/2) / (c0*dt)
                
# Edge Fits
left_fit = np.polyfit(td.time.values[:Npoints], td.data.values[:Npoints], 1)
right_fit = np.polyfit(td.time.values[-Npoints:], td.data.values[-Npoints:], 1)
start_time = td.time.mean() - reduce_df/2*(td.time.max()-td.time.min())
end_time = td.time.mean() + reduce_df/2*(td.time.max()-td.time.min())
start_value = np.polyval(left_fit, start_time)
end_value = np.polyval(right_fit, end_time)

# Subtract linear component
linearfit = np.polyfit(range(len(data_values)), data_values,1)
data_values = data_values - np.polyval(linearfit, range(len(data_values)))