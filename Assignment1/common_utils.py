### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def extract_features(filepath):
    
    '''
    Source: https://github.com/danz1ka19/Music-Emotion-Recognition/blob/master/Feature-Extraction.py
    Modified to process a single file

        function: extract_features
        input: path to the audio file
        output: csv file containing features extracted

        This function reads the content in a directory and for each audio file detected
        reads the file and extracts relevant features using librosa library for audio
        signal processing
    '''
    n_fft = 2048

    feature_set = {}  # Features

    # Reading audio file
    y, sr = librosa.load(filepath)
    S = np.abs(librosa.stft(y, n_fft=n_fft)) 
    # https://librosa.org/doc/main/generated/librosa.stft.html (set 512 for speech processing)

    # Extracting Features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    rmse = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
    poly_features = librosa.feature.poly_features(S=S, sr=sr, n_fft=n_fft)
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, )
    mfcc_delta = librosa.feature.delta(mfcc)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    # Concatenating Features into one csv and json format
    feature_set['filename'] = [os.path.basename(filepath)]  # song name
    feature_set['tempo'] = [tempo[0]]  # tempo 
    # try:
    #     feature_set['tempo'] = [tempo[0]]  # tempo 
    # except:
    #     return None
    feature_set['total_beats'] = [sum(beats)]  # beats
    feature_set['average_beats'] = [np.average(beats)]
    feature_set['chroma_stft_mean'] = [np.mean(chroma_stft)]  # chroma stft
    feature_set['chroma_stft_var'] = [np.var(chroma_stft)]
    
    feature_set['chroma_cq_mean'] = [np.mean(chroma_cq)]  # chroma cq
    feature_set['chroma_cq_var'] = [np.var(chroma_cq)]
    
    feature_set['chroma_cens_mean'] = [np.mean(chroma_cens)]  # chroma cens
    feature_set['chroma_cens_var'] = [np.var(chroma_cens)]
    feature_set['melspectrogram_mean'] = [np.mean(melspectrogram)]  # melspectrogram
    feature_set['melspectrogram_var'] = [np.var(melspectrogram)]
    feature_set['mfcc_mean'] = [np.mean(mfcc)]  # mfcc
    feature_set['mfcc_var'] = [np.var(mfcc)]
    feature_set['mfcc_delta_mean'] = [np.mean(mfcc_delta)]  # mfcc delta
    feature_set['mfcc_delta_var'] = [np.var(mfcc_delta)]
    feature_set['rmse_mean'] = [np.mean(rmse)]  # rmse
    feature_set['rmse_var'] = [np.var(rmse)]
    feature_set['cent_mean'] = [np.mean(cent)]  # cent
    feature_set['cent_var'] = [np.var(cent)]
    feature_set['spec_bw_mean'] = [np.mean(spec_bw)]  # spectral bandwidth
    feature_set['spec_bw_var'] = [np.var(spec_bw)]
    feature_set['contrast_mean'] = [np.mean(contrast)]  # contrast
    feature_set['contrast_var'] = [np.var(contrast)]
    feature_set['rolloff_mean'] = [np.mean(rolloff)]  # rolloff
    feature_set['rolloff_var'] = [np.mean(rolloff)]
    feature_set['poly_mean'] = [np.mean(poly_features)]  # poly features
    feature_set['poly_var'] = [np.mean(poly_features)]
    
    feature_set['tonnetz_mean'] = [np.mean(tonnetz)]  # tonnetz
    feature_set['tonnetz_var'] = [np.var(tonnetz)]
    
    feature_set['zcr_mean'] = [np.mean(zcr)]  # zero crossing rate
    feature_set['zcr_var'] = [np.var(zcr)]
    feature_set['harm_mean'] = [np.mean(harmonic)]  # harmonic
    feature_set['harm_var'] = [np.var(harmonic)]
    feature_set['perc_mean'] = [np.mean(percussive)]  # percussive
    feature_set['perc_var'] = [np.var(percussive)]
    feature_set['frame_mean'] = [np.mean(frames_to_time)]  # frames
    feature_set['frame_var'] = [np.var(frames_to_time)]
    
    for ix, coeff in enumerate(mfcc):
        feature_set['mfcc' + str(ix) + '_mean'] = [coeff.mean()]
        feature_set['mfcc' + str(ix) + '_var'] = [coeff.var()]
    
    return feature_set


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

#### ADD ADDITIONAL FUNCTIONS HERE AFTER COMPLETING QUESTION A1 ####

class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels, drop_out):
        super(MLP, self).__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits