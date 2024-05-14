# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:37:21 2024

Real-time calculation and plotting of PSD
------------------------------------------
The script reads the LSL stream in real-time and computes the PSD and plots 
the PSD plot.

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de

"""

# SET PARAMS -----------------------------------------

# duration of the epoch (in secs)
duration = 4 
# target event markers
targetEvents = ['13', '14', '15', '16']
# channel to plot the psd
chan2plot = 22

# ----------------------------------------------------


# -- libraires
import numpy as np
import matplotlib.pyplot as plt
# for LSL communication
from pylsl import StreamInlet, resolve_stream
from scipy.signal import welch

# -- functions
# function to find an LSL stream
def findStream(stream_type, stream_name=None):
    streams = resolve_stream('type', stream_type)
    if stream_name:
        for stream in streams:
            if stream.name() == stream_name:
                return stream
    return streams[0] if streams else None

# function to compute PSD
def computePSD(data):
    # data shape
    ntrls, nsamps, nchans = data.shape 
    window_length = nsamps
    overlap = window_length / 2
    # calculating nfft 
    nfft = 2**(np.ceil(np.log2(nsamps)).astype(int))
    # freq resolution 
    nfreqs = nfft // 2 + 1
    # empty matrix to store psd values
    trialPSD = np.zeros((ntrls,nchans, nfreqs))
    # loop over trials 
    for itrl in range(ntrls):
        # loop over channels
        for ichan in range(nchans):
            # calculate PSD 
            psdfreqs, psd = welch(data[itrl,:,ichan], fs=srate, window='hamming', nperseg=window_length, noverlap=overlap, nfft=nfft)
            trialPSD[itrl, ichan, :] = psd.ravel()
    return trialPSD, psdfreqs


# -- connecting to LSL stream
# # eeg stream
print("Looking for an EEG stream...")
eegStream = findStream('EEG', 'SSVEPstream')
# marker stream
print("Looking for an Marker stream...")
markerStream = findStream('Markers', 'Markers')

# create StreamInlets if streams were found
# eeg stream
if eegStream:
    eegInlet = StreamInlet(eegStream)
    print("Connected to EEG stream.")
else:
    print("EEG stream not found.")
    eegInlet = None
# marker stream
if markerStream:
    markerInlet = StreamInlet(markerStream)
    print("Connected to Marker stream.")
else:
    print("Marker stream not found.")
    markerInlet = None

# sampling rate of EEG
srate = eegInlet.info().nominal_srate()
print(f'Sampling rate: {srate}')
# number of channels 
nbchans =  eegInlet.info().channel_count()

# time points in single epoch 
samples = int(srate * duration)
tpts = samples
# to store all epochs  
epochData = [] 

# setting up a plot
fig, ax = plt.subplots()

# -- extracting the data 
while eegInlet or markerInlet:
    # check if eeg stream is not empty
    if eegInlet:
        # collect the eeg data
        eeg_sample, timestamp = eegInlet.pull_sample(timeout=0.0)
    # check if marker stream is not empty
    if markerInlet:
        # collect the markers
        marker, timestamp = markerInlet.pull_sample(timeout=0.0)
        
    # check for markers are not empty
    if marker:
        # check for stim markers     
        if marker[0] in targetEvents: 
            print('Stim detected, processing eeg data......')
            # Reset buffer when a new epoch starts
            buffer = np.zeros((samples, nbchans))
            # loop over length of epochs
            for i in range(samples):
                # collect eeg data for each epochs
                eegSample, timestamp = eegInlet.pull_sample(timeout=1.0)
                # store the eeg data
                buffer[i, :] = eegSample 
            # store the new epoch to the epochs 
            epochData.append(buffer)
            psdvals, psdfreqs = computePSD(np.stack(epochData, axis=0))
            
            # setting up the plot
            ax.clear()
            ax.plot(psdfreqs, np.mean(psdvals[:,chan2plot,:], axis=0))
            ax.set_xlim(5,45)
            ax.set_title('Power Spectral Density')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('power [a.u]')
            plt.draw()
            plt.pause(0.01)
            
plt.show()            

                