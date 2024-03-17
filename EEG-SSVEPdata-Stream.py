#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:14:43 2024

The script streams the EEG data recorded during an SSVEP experiment. 
original code from: https://github.com/chkothe/pylsl/blob/master/examples/SendDataAdvanced.py

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

# libraries 
import mne
import time
import os.path as op
from pylsl import StreamInfo, StreamOutlet, local_clock

# load the data
rootpath = r'/Users/abinjacob/Documents/01. Calypso/SSVEP OnlineClassify/data'
# EEGLab file to load (.set)
filename = 'P02_SSVEP_raw24Chans.set'
filepath = op.join(rootpath,filename)
# load file in mne 
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)

# extract data and times
data, times = raw[:, :]
# sampling frequency 
sfreq = raw.info['sfreq']
# channel names 
chnames = raw.info['ch_names']
nchans = len(chnames)

# extract event markers 
events = mne.events_from_annotations(raw)
event_ids = events[0][:,2]
event_timestamps = events[0][:,0]

# create EEG StreamInfo 
eeg_info = StreamInfo('SSVEPstream', 'EEG', nchans, sfreq, 'float32', 'myuid4242')

# create marker StreamInfo
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'myuid4243')

# Add channel information
channels = eeg_info.desc().append_child("channels")
for ch_name in chnames:
    channels.append_child("channel")\
             .append_child_value("name", ch_name)\
             .append_child_value("unit", "microvolts")\
             .append_child_value("type", "EEG")

# Initialize the StreamOutlet
eeg_outlet = StreamOutlet(eeg_info)
marker_outlet = StreamOutlet(marker_info)
print("Now streaming SSVEP data and Markers...")

# stream the data and markers
for i in range(data.shape[1]):
    # stream EEG sample
    sample = data[:, i].tolist()
    eeg_outlet.push_sample(sample, local_clock())
    
    # check if the current sample corresponds to an event marker
    if i in event_timestamps:
        # find the event ID(s) corresponding to the current timestamp
        event_id = [str(event_ids[j]) for j in range(len(event_ids)) if event_timestamps[j] == i]
        # stream the marker
        for eid in event_id:
            marker_outlet.push_sample([eid], local_clock())
            print(f'{eid}')
    
    # wait before sending the next sample to mimic real-time streaming
    time.sleep(1.0 / sfreq)


