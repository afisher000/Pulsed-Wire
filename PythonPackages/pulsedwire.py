# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:51:47 2022

@author: afisher
"""
import matplotlib.pyplot as plt
import numpy as np


def annotated_plot(df, TRAJ, cut):
    '''Create a plot showing time cuts.'''
    fig, ax = plt.subplots()
    ax.plot(df.time, df[TRAJ], label='signal')
    ax.plot(df.time[cut=='ref1'], df[TRAJ][cut=='ref1'], label='ref1')
    ax.plot(df.time[cut=='ref2'], df[TRAJ][cut=='ref2'], label='ref2')
    ax.plot(df.time[cut=='pks'], df[TRAJ][cut=='pks'], label='pks')
    plt.legend()
    return

def poly_minmax(signal, left_idx, right_idx, fcn):
    '''Create polynomial fit of data near peak. 
    Return min/max as given by fcn arg.'''
    porder = 5
    nearpeak = signal.loc[left_idx:right_idx].copy() # Data near peak
    nearpeak['ctime'] = nearpeak.time - nearpeak.time.mean()
    poly = np.polyfit(nearpeak.ctime, nearpeak.volts, porder) 
    
    # Troubleshooting figure
    #fig, ax = plt.subplots()
    #ax.plot(nearpeak.time, nearpeak.volts)
    #ax.plot(nearpeak.time, np.polyval(poly, nearpeak.ctime))
    
    return fcn(np.polyval(poly, nearpeak.ctime))


def get_amplitudes(signal, peaks):
    '''Compute amplitudes at all extrema'''
    period = np.diff(peaks).mean()
    halfperiod = round(period/2)
    step = round(period/10)
    
    # Find top peaks
    top_peaks = np.array([poly_minmax(signal, peak-step, peak+step, max)
                 for peak in peaks])
    bot_peaks = np.array([poly_minmax(signal, peak-step, peak+step, min)
                 for peak in peaks[:-1]+halfperiod])
    
    # Compute average pk2pk amplitude at each extrema
    t = top_peaks.repeat(2)/2
    b = bot_peaks.repeat(2)/2
    amplitudes = t[1:-2] + t[2:-1] - b[:-1] - b[1:]
    return amplitudes

def fit_concavity(series):
    '''Return the axis and offset of a poly fit'''
    a,b,c = np.polyfit(series.index, series.values, 2)
    axis = -b/(2*a)
    offset = c-b**2/(4*a)
    return axis, offset

def plot_concavity(df, qfit_axes, CONCAVITY, idxs):
    '''Plot concavities for peaks against theory'''
    fig, ax = plt.subplots()
    for idx in idxs:
        plt.scatter(df.index, df.iloc[:,idx], label=f'Peak {idx}')
    avg_axis = qfit_axes[idxs].mean()
    offsets = np.linspace(-1500,1500,100)
    ku = 2*np.pi/32e3 #um
    alpha = 1.53
    beta = 1.16
    if CONCAVITY:
        plt.plot(offsets, 1+0.5*(ku*alpha*(offsets-avg_axis))**2, label='Theory')
    else:
        plt.plot(offsets, 1-0.5*(ku*beta*(offsets-avg_axis))**2, label='Theory')
    plt.legend()
    return

def plot_axis_along_undulator(qfit_axes):
    '''Plot estimate axis along undulator'''
    fig, ax = plt.subplots()
    ax.plot(qfit_axes)
    ax.set_xlabel('Peak Number')
    ax.set_ylabel('Xoffset (um)')
    ax.set_title('Plotting axis of symmetry by peak')
    return