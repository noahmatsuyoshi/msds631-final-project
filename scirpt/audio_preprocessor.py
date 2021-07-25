import sys
import os
import numpy as np
import time
from pathlib import Path
from matplotlib import pyplot as plt

import librosa
from librosa.display import specshow
from parameter_input import *

test_path = sys.argv[1] # input wav file path

### preprocessor 

class Audio_Preprocessor:
    """
    audio preprocessor
    
    transform_to_Sdb: transform audio into numpy array with two channels
    
    slice_train: slice train audio array into fixed size window
    
    slice_test: slice test audio array into fixed size window
    
    
    """

    def __init__(self,
                 hops_per_sec=HOPS_PER_SEC,
                 sp_rt=SAMPLING_RATE,
                 hop_len=int(SAMPLING_RATE / HOPS_PER_SEC),
                 n_mels=N_MELS,
                 offset=OFFSET,
                 duration=WIN_SEC):
        self.hops_per_sec = hops_per_sec
        self.sp_rt = sp_rt
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.offset = offset
        self.duration = duration
        self.width = self.duration * self.hops_per_sec

    def transform_to_Sdb(self, aud_file):

        self.stem = Path(aud_file).stem

        mp3, sr = librosa.load(aud_file, sr=self.sp_rt, mono = False)
        total_time = librosa.samples_to_time(len(mp3), self.sp_rt)
        
        S0 = librosa.feature.melspectrogram(
            y=mp3[0,:], sr=self.sp_rt, n_mels=self.n_mels, hop_length=self.hop_len)
        S1 = librosa.feature.melspectrogram(
            y=mp3[1,:], sr=self.sp_rt, n_mels=self.n_mels, hop_length=self.hop_len)
        S = np.dstack([S0, S1])

        self.S_db = librosa.power_to_db(S, ref=np.max)
        self.S_db = np.swapaxes(self.S_db, 0, 1)
        
            
        print(f'audio {self.stem} total {total_time} seconds, with S_db shape {self.S_db.shape}')
        
    
    def slice_train(self, save = True):
        S_db_scale = self.S_db/-80  # normalize into range [0, 1]
        n, m, k = S_db_scale.shape  # m = N_MELS
        
        resx_train = []
        for i in range(0, n - self.width + 1, 5): # 5 hops ahead overlapping
            X = S_db_scale[i:self.width + i, :, :]
            resx_train.append(X)
            
        self.train_array = np.stack(resx_train)
        
        if save:
            np.save(f'npy/{self.stem}.npy', self.train_array)
            
        print(f'split audio {self.stem}.wav for training with shape: {self.train_array.shape}')
        
    
    def slice_test(self):

        S_db_scale = self.S_db/-80 # normalize into range [0, 1]
        n, m, k = S_db_scale.shape  # m = N_MELS
        
        resx_test = []
        for i in range(0, n, self.width): # non-overlapping for test
            x = S_db_scale[i:i+self.width, :, :]
            left = self.width - x.shape[0]
            if left > 0:
                x = S_db_scale[i-left:, :, :]
            resx_test.append(x)
            
        self.test_array = np.stack(resx_test)
        print(f'split audio {self.stem}.wav for testing with shape: {self.test_array.shape}')
        
        
## save training audio file as npy array

ap = Audio_Preprocessor()
ap.transform_to_Sdb(test_path)
ap.slice_train(save = True) 







