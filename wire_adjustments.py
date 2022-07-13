# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:55:55 2022

@author: afisher
"""
import numpy as np
import matplotlib.pyplot as plt


## Longitudinal distance in meters, transverse displacement in um
# Desire movement of wire at firstpk and lastpk
pk_dx = (600, 800)
pk_dy = (0, -200)


# Define z=0 at P1
p1_to_und = 39*0.0254
und_to_p2 = 5.5*0.0254
und_length = .35 #Include endplates
post_z = (0, p1_to_und + und_length + und_to_p2)

num_peaks = 15
dz = (und_length - (num_peaks-1)*.032/2) / 2
pk_z = (p1_to_und + dz, p1_to_und + und_length - dz)

# Fit line through peak adjustments 
xfit = np.polyfit(pk_z, pk_dx, 1)
post_dx = np.polyval(xfit, post_z)

yfit = np.polyfit(pk_z, pk_dy, 1)
post_dy = np.polyval(yfit, post_z)

# Plot as check
fig, ax = plt.subplots()
ax.scatter(pk_z, pk_dx, label='Axis for peaks')
ax.plot(post_z, post_dx, label='Linear Fit')
ax.legend()
ax.set_xlabel('Z (m)')
ax.set_ylabel('X (um)')


fig, ax = plt.subplots()
ax.scatter(pk_z, pk_dy, label='Axis for peaks')
ax.plot(post_z, post_dy, label='Linear Fit')
ax.legend()
ax.set_xlabel('Z (m)')
ax.set_ylabel('Y (um)')





