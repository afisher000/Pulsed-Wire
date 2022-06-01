# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:51:47 2022

@author: afisher
"""
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


def get_amplitudes_as_series(df, bins, num_peaks=19, noise_fac=2):
    '''Compute amplitudes of signal region and return as a pandas Series object.'''
    
    # Find peaks
    cut = pd.cut(df.time, bins.values, labels=bins.index[:-1])
    signal = df.loc[cut=='pks'].reset_index(drop=True)
    noise = df.volts[cut=='ref1'].max() - df.volts[cut=='ref1'].min()
    peaks = np.append(find_peaks(signal.volts, prominence=noise*noise_fac, distance=200)[0],
                     find_peaks(-1*signal.volts, prominence=noise*noise_fac, distance=200)[0])
    if len(peaks)!=num_peaks:
        #raise Exception(f'Incorrect number of peaks (found {len(peaks)}, not {num_peaks})')
        return
            
    # Compute amplitudes 
    ref_amp = abs(df.volts[cut=='ref2'].mean() - df.volts[cut=='ref1'].mean())
    amps = compute_amplitudes(signal, peaks)/ref_amp
    return pd.Series(amps)

def get_offset_from_filename(data_folder, file):
    '''Parse filename and return wire offset.'''
    regex = re.compile('\(([-?\d]+),([-?\d]+)\).([\d]+).csv')
    xoffset, yoffset, duplicate_id = map(int, regex.findall(file)[0])
    offset = xoffset if 'xoffset' in data_folder else yoffset
    return offset

    
def parse_global_constants(data_folder):
    '''Interpret global constants from data_folder name.'''
    TRAJ = 'y' if 'ytraj' in data_folder else 'x'
    OFF = 'yoffset' if 'yoffset' in data_folder else 'xoffset'
    CONCAVITY = 1 if TRAJ[0]!=OFF[0] else 0
    GBL = pd.Series([TRAJ, OFF, CONCAVITY], index=['TRAJ','OFF','CONCAVITY'])
    return GBL

def annotated_plot(df, bins, title=''):
    '''Create a plot showing time cuts.'''
    cut = pd.cut(df.time, bins.values, labels=bins.index[:-1])
    
    fig, ax = plt.subplots()
    ax.plot(df.time, df.volts, label='signal')
    ax.plot(df.time[cut=='ref1'], df.volts[cut=='ref1'], label='ref1')
    ax.plot(df.time[cut=='ref2'], df.volts[cut=='ref2'], label='ref2')
    ax.plot(df.time[cut=='pks'], df.volts[cut=='pks'], label='pks')
    ax.legend()
    ax.set_title(title)
    return

def poly_minmax(signal, center, width, porder=5, plot=False):
    '''Create polynomial fit of inputs. Return min/max.'''
    nearpeak = signal.loc[(center-width):(center+width)].copy()
    nearpeak['ctime'] = nearpeak.time - nearpeak.time.mean() #ctime = centered time
    poly = np.polyfit(nearpeak.ctime, nearpeak.volts, porder) #Demean
    polyvals = np.polyval(poly, nearpeak.ctime)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(nearpeak.time, nearpeak.volts)
        ax.plot(nearpeak.time, polyvals)
    
    if poly[3]>0: #Sign of quadratic term
        return min(polyvals)
    else:
        return max(polyvals)
    

def compute_amplitudes(signal, peaks):
    '''Compute amplitudes at all extrema'''
    period = np.diff(peaks).mean()
    width = round(period/5)
    peaks.sort() #Make sure sorted
    
    # Find peak values
    peak_vals = np.array([poly_minmax(signal, peak, width)
                 for peak in peaks])

    # Drop first and last amplitude where mean cannot be calculated
    amplitudes = np.abs(np.convolve(peak_vals, [0.5, -1, 0.5], mode='valid'))[1:-1]
    means   = np.convolve(peak_vals, [.25, .5, .25], mode='valid')
    deltay = means[2:]-means[:-2] #deltay = change in mean over period

    # Apply measurement correction (to account for strong linear chirp)
    delta = 2*deltay/amplitudes
    meas_error = 3e-5*delta**4 + 1.27e-2*delta**2
    amplitudes = amplitudes*(1-meas_error)
    
    return amplitudes

def compare_trajectories(hp, pw, bins, noise_fac=2, save=False, file=None):
    '''Compares the fit of the pulse wire trajectory to the integrated
    hall probe trajectory.'''

    # Find pw_peaks
    cut = pd.cut(pw.time, bins.values, labels=bins.index[:-1])
    signal = pw.loc[cut=='pks'].reset_index()
    noise = pw.volts[cut=='ref1'].max() - pw.volts[cut=='ref1'].min()
    pw_peaks = np.append(find_peaks(signal.volts, prominence=noise*noise_fac, distance=200)[0],
                         find_peaks(-1*signal.volts, prominence=noise*noise_fac, distance=200)[0])
        
    # Find hp_peaks
    noise = (hp.volts.max() - hp.volts.min())/5
    hp_peaks = np.append(find_peaks(hp.volts, prominence=noise, distance=10)[0],
                         find_peaks(-1*hp.volts, prominence=noise, distance=10)[0])
    if len(hp_peaks)!=len(pw_peaks):
        raise Exception('Found {len(hp_peaks) hall probe peaks but {len(pw_peaks)} pulsed wire peaks.')
        
    # Fit linear regression for time
    xmodel = LinearRegression()
    xmodel.fit(signal.time[pw_peaks].values.reshape(-1,1), 
              hp.time[hp_peaks].values)
    pw['pred_time'] = xmodel.predict(pw.time.values.reshape(-1,1))
    
    # Fit linear regression for volts
    ymodel = LinearRegression()
    ymodel.fit( np.stack([signal.volts[pw_peaks].values,
                         hp.time[hp_peaks].values]).T,
               hp.volts[hp_peaks].values)
    pw['pred_volts'] = ymodel.predict( np.stack([pw.volts.values,
                                                 pw.pred_time]).T)
    
    # Interpolate pw
    hp['pw_interp'] = interp1d(pw.pred_time, pw.pred_volts)(hp.time)
    
    # Get filenames
    if file is None:
        filename = f'HP-PW Comparion.jpg'
    else:
        filename = file
        
    # Plot fitted trajectories
    fig, ax = plt.subplots()
    ax.plot(pw.pred_time, pw.pred_volts, label='Pulsed Wire (scaled)')
    ax.plot(hp.time, hp.volts, label='Hall Probe')
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('Trajectories (m)')
    ax.set_title('Comparing Trajectories')
    ax.legend()
    ax.set_title(filename[:-3])
    if save:
        plt.savefig(filename)
    
    # Plot error
    fig, ax = plt.subplots()
    ax.plot(hp.time, hp.volts-hp.pw_interp)
    ax.plot(hp.time[hp_peaks], hp_peaks*0, c='k')
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('Error')
    ax.set_title(filename[:-3])
    if save:
        plt.savefig('Error of '+file)
            
    qfit = np.polyfit(hp.time, hp.volts-hp.pw_interp,2)
    fig, ax = plt.subplots()
    ax.plot(hp.time, hp.volts-hp.pw_interp)
    ax.plot(hp.time, np.polyval(qfit, hp.time))
    
    return

def fit_concavity(series):
    '''Return the axis and offset of a poly fit'''
    a,b,c = np.polyfit(series.index, series.values, 2)
    axis = -b/(2*a)
    offset = c-b**2/(4*a)
    return axis, offset

def plot_concavity(df, GBL, qfit_axes, pk_idxs=range(3)):
    '''Plot concavities for peaks against theory'''
    fig, ax = plt.subplots()
    for idx in pk_idxs:
        ax.scatter(df.index, df.iloc[:,idx], label=f'Peak {idx}')
    avg_axis = qfit_axes[pk_idxs].mean()
    
    offsets = np.linspace(-1500,1500,100)
    ku = 2*np.pi/32e3 #um
    alpha = 1.53
    beta = 1.16
    if GBL.CONCAVITY:
        ax.plot(offsets, 1+0.5*(ku*alpha*(offsets-avg_axis))**2, label='Theory')
    else:
        ax.plot(offsets, 1-0.5*(ku*beta*(offsets-avg_axis))**2, label='Theory')
    ax.legend()
    ax.set_xlabel(f'{GBL.OFF} (um)')
    ax.set_ylabel('Normalized Amp.')
    
    if pk_idxs[0]>=0:
        ax.set_title('Entrance Peaks')
    else:
        ax.set_title('Exit Peaks')
    return

def plot_axis_along_undulator(GBL, qfit_axes, save=False):
    '''Plot estimate axis along undulator'''
    fig, ax = plt.subplots()
    ax.plot(qfit_axes)
    ax.set_xlabel('Peak Number')
    ax.set_ylabel(GBL.OFF+' um')
    ax.set_title(f'Offset along axis from {GBL.TRAJ}traj data')
    if save:
        plt.savefig(f'{GBL.OFF} predicted with {GBL.TRAJ}traj.jpg')
    
    
    return