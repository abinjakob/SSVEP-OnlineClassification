# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:49:27 2024

SSVEP Online SVM Classifier using CCA Features (timing test)
-----------------------------------------------
Feature used: CCA Correlation Values for stimulus frequencies and its harmonics

Classification: SVM classifier with 5-Fold crossvalidation
                - spliting data using train_test_split
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV
                
Model will be saved as 'SVMclassifier_ccaSSVEP.joblib'

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de

"""


# SET PARAMS -----------------------------------------

# duration of the epoch
duration = 4 # in seconds
# string to detect stim period
trigger = 'stim'
# stimulation frequencies
freqs = [15, 20]
srate = 500
# target event markers
targetEvents = [13, 14, 15, 16]
nbchans = 24

# ----------------------------------------------------


# -- libraires
import mne
import numpy as np
import os.path as op
# SVM classifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import load
# CCA
from sklearn.cross_decomposition import CCA
# for LSL communication
from pylsl import StreamInlet, resolve_stream
import time

# -- functions
# function to find an LSL stream
def find_stream(stream_type, stream_name=None):
    streams = resolve_stream('type', stream_type)
    if stream_name:
        for stream in streams:
            if stream.name() == stream_name:
                return stream
    return streams[0] if streams else None

# function to compute CCA
def computeCCA(epoch):
    # eeg data from the epocs 
    X_data = epoch.T
    # generating time vector
    t = np.linspace(0, duration, tpts, endpoint= False)
    CCAfeatures = []

    # loop over frequencies
    for i, iFreq in enumerate(freqs):    
        # create the sine and cosine signals for 1st harmonics
        sine1 = np.sin(2 * np.pi * iFreq * t)
        cos1 = np.cos(2 * np.pi * iFreq * t)
        # create the sine and cosine signals for 2nd harmonics
        sine2 = np.sin(2 * np.pi * (2 * iFreq) * t)
        cos2 = np.cos(2 * np.pi * (2 * iFreq) * t)
        
        # create Y vector 
        Y_data = np.column_stack((sine1, cos1, sine2, cos2))
        
        # performing CCA
        # considering the first canonical variables
        cca = CCA(n_components= 1)
        # compute cannonical variables
        cca.fit(X_data, Y_data)
        # return canonical variables
        Xc, Yc = cca.transform(X_data, Y_data)
        corr = np.corrcoef(Xc.T, Yc.T)[0,1]
        
        # store corr values for current epoch
        CCAfeatures.append(corr)
        
    return np.array(CCAfeatures).reshape(1, 2)
    
# function for classification
def SVMclassifier(epoch):
    t1 = time.time()
    X = computeCCA(epoch)
    pred = clf.predict(X)
    t2 = time.time()
    ctime = t2-t1
    return pred, ctime



# -- connecting to LSL stream
# # eeg stream
print("Looking for an EEG stream...")
eegStream = find_stream('EEG', 'SSVEPstream')
# marker stream
print("Looking for an Marker stream...")
markerStream = find_stream('Markers', 'Markers')

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


# time points in single epoch 
samples = int(srate * duration)
tpts = samples
 
# -- load the SVM-CCA trained model
modelpath = r'L:\Cloud\Calypso\Online Classification'
modelname = 'SVMclassifier_ccaSSVEPnew.joblib'
modelimport = op.join(modelpath,modelname)
# create instance of the model 
clf = load(modelimport)

accuracy = []
predtime = []

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
        # converting marker to int
        marker_int = int(marker[0])
        # check for stim markers     
        if marker_int in targetEvents: 
            print('Stim detected, processing eeg data......')
            # Reset buffer when a new epoch starts
            buffer = np.zeros((samples, nbchans))
            # loop over length of empochs
            for i in range(samples):
                # collect eeg data for each epochs
                eegSample, timestamp = eegInlet.pull_sample(timeout=1.0)
                # store the eeg data
                buffer[i, :] = eegSample 
    
            # running the svm classifier
            result, timetaken = SVMclassifier(buffer.T)
            if marker_int in [13,15]:
                print(f'Prediction = {result} Label = 15')
                label = 15
            elif marker_int in [14,16]:
                print(f'Prediction = {result} Label = 20')
                label = 20
            # record prediction time 
            predtime.append(timetaken)
            # record accuracy 
            if label == result:
                accuracy.append(1)
                
            acc = (sum(accuracy)/len(predtime))*100
            meanpretime = np.mean(predtime) 
            print(f'Prediction Time: {timetaken*1000:.2f}ms, Avg. Prediction Time: {meanpretime*1000:.2f}ms, Avg. Accuracy: {acc}, Trials: {len(predtime)}')
            

        
#%% plotting prediction time    
import matplotlib.pyplot as plt
predtime_ms = [t*1000 for t in predtime]
plt.plot(predtime_ms, 'b*')
plt.xlabel('trials')
plt.ylabel('prediction time [ms]')
plt.xlim([0, len(predtime_ms)])
plt.ylim([0, 50])
plt.title(f'Online Prediction Time in Windows PC (Avg Pred Time= {meanpretime*1000:.2f}ms)')



