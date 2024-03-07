# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:39:14 2024

SSVEP Online SVM Classifier using CCA Features 
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

@author: togo2120
"""


# SET PARAMS -----------------------------------------

# duration of the epoch
duration = 4.2 # in seconds
# string to detect stim period
trigger = 'stim'
# stimulation frequencies
freqs = [15, 20]
srate = 500

# ----------------------------------------------------


# libraires
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


# # connecting to LSL stream 
# # printing status message
# print("Looking for an EEG stream...")
# # resolve and initialise EEG stream using LSL
# streams = resolve_stream('type', 'EEG')  
# inlet = StreamInlet(streams[0])
# # retriving sampling rate from stream info
# # srate = inlet.info().nominal_srate()
# print(f'Sampling rate: {srate}')
tpts = 2101
 
# load the SVM-CCA trained model
modelpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Python\Models'
modelname = 'SVMclassifier_ccaSSVEP.joblib'
modelimport = op.join(modelpath,modelname)
# create instance of the model 
clf = load(modelimport)


# # function to 
# def processData():
#     global epochs
#     sample, timestamp = inlet.pull_sample()
#     # check if event markers indicate a stim
#     if eventmark == 

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
    X = computeCCA(epoch)
    pred = clf.predict(X)
    return pred
      
#%% parameters

# filter
l_freq = 0.1 
h_freq = None
# epoching 
tmin = -0.2
tmax = 4
# Events
event_id = {'stim_L15': 13, 'stim_L20': 14, 'stim_R15': 15, 'stim_R20': 16}
event_names = list(event_id.keys())
foi = [15, 20, 15, 20] # Freqs of interest
# load data 
rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'
# EEGLab file to load (.set)
filename = 'P02_SSVEP_raw24Chans.set'
filepath = op.join(rootpath,filename)
# load file in mne 
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)
a = raw.info
#Preprocess the data
# extracting events 
events, _ = mne.events_from_annotations(raw, verbose= False)
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['stim_L15'], event_id['stim_L20'], event_id['stim_R15'], event_id['stim_R20']], 
    tmin=tmin, tmax=tmax, 
    baseline= None, 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 3.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)


#%% model predict

iTrial = 1;
eegSignal = epochs.get_data()
epoch = eegSignal[iTrial,:,:]
pred = SVMclassifier(epoch)
        
    
    



