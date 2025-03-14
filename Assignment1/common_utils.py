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
    This function reads the content in a directory and for each audio file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
    '''
    # Reading audio file
    y, sr = librosa.load(filepath, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.harmonic(y), librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    features = [
                f"{filepath}",
                np.mean(chroma_stft), np.var(chroma_stft),
                np.mean(rms), np.var(rms),
                np.mean(spec_cent), np.var(spec_cent),
                np.mean(spec_bw), np.var(spec_bw),
                np.mean(rolloff), np.var(rolloff),
                np.mean(zcr), np.var(zcr),
                np.mean(harmony), np.var(harmony),
                np.mean(perceptr), np.var(perceptr),
                float(tempo)
            ]

    for coeff in mfcc:
        features.append(np.mean(coeff))
        features.append(np.var(coeff))

    columns=['filename',
         'chroma_stft_mean', 'chroma_stft_var',
         'rms_mean', 'rms_var',
         'spectral_centroid_mean', 'spectral_centroid_var',
         'spectral_bandwidth_mean', 'spectral_bandwidth_var',
         'rolloff_mean', 'rolloff_var',
         'zero_crossing_rate_mean','zero_crossing_rate_var',
         'harmony_mean', 'harmony_var',
         'perceptr_mean', 'perceptr_var',
         'tempo'] + \
         [f'mfcc{i+1}_{stat}' for i in range(20) for stat in ['mean', 'var']]

    feature_set = pd.DataFrame([features], columns=columns)
         
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


class MLP(nn.Module):
    def __init__(self, no_features, no_hidden1, drop_out, no_hidden2=128, no_hidden3=128, no_labels=1):
        super(MLP, self).__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden1),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden1, no_hidden2),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden2, no_hidden3),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(no_hidden3, no_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits
    
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((pred > 0.5).float() == y).all(dim=1).sum().item()
    
    train_loss /= size
    train_correct /= size

    return train_loss, train_correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            test_correct += ((pred > 0.5).float() == y).all(dim=1).sum().item()

    test_loss /= size
    test_correct /= size

    return test_loss, test_correct