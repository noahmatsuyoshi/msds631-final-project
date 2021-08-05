import sys
import os
import numpy as np
import time
from pathlib import Path
from matplotlib import pyplot as plt

import librosa
from librosa.display import specshow
import pandas as pd
from parameter_input import *

metadata_df=pd.read_csv('../tiny_data/UrbanSound8K.csv')

class Audio_Preprocessor:
    """
    audio preprocessor

    transform_to_Sdb: transform audio into numpy array with two channels

    slice_train: slice train audio array into fixed size window

    slice_test: slice test audio array into fixed size window


    """

    def __init__(self,
                 hops_per_sec=HOPS_PER_SEC,
                 hop_len=int(SAMPLING_RATE / HOPS_PER_SEC),
                 n_mels=N_MELS,
                 offset=OFFSET,
                 duration=WIN_SEC):
        self.hops_per_sec = hops_per_sec
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.offset = offset
        self.duration = duration
        self.width = self.duration * self.hops_per_sec
 
    def check(self, aud_file):

        self.stem = Path(aud_file).stem
        sr = librosa.core.get_samplerate(aud_file)

        if sr != 44100:
            self.skip = True
        else:
            self.skip = False
            self.sp_rt = 44100
            
            
        self.mp3, _ = librosa.load(aud_file, sr=self.sp_rt, mono=False)
        total_time = librosa.samples_to_time(self.mp3.shape[-1], self.sp_rt)
        
        if total_time < 1 or len(self.mp3.shape) != 2:
            self.skip = True
        
        #print(f'audio {self.stem} total {total_time} seconds.')

    def transform_to_Sdb(self):
        S0 = librosa.feature.melspectrogram(
            y=self.mp3[0, :], sr=self.sp_rt, n_mels=self.n_mels, hop_length=self.hop_len)
        S1 = librosa.feature.melspectrogram(
            y=self.mp3[1, :], sr=self.sp_rt, n_mels=self.n_mels, hop_length=self.hop_len)
        S = np.dstack([S0, S1])

        self.S_db = librosa.power_to_db(S, ref=np.max)
        self.S_db = np.swapaxes(self.S_db, 0, 1)

    def slice_train(self):
        S_db_scale = self.S_db/-80  # normalize into range [0, 1]
        n, m, k = S_db_scale.shape  # m = N_MELS

        self.resx_train = {}
        for i in range(0, n - self.width + 1, 10):  # 5 hops ahead overlapping
            X = S_db_scale[i:self.width + i, :, :]
            #self.resx_train.append(X)
            filename = f'{self.stem}_w{i}.npy'
            self.resx_train[filename] = X

        print(f'split {self.stem}.wav for training into array with {len(self.resx_train)} windows')
        
        
ap = Audio_Preprocessor()

slice_path = []
slice_classid = []
#for i in range(1, 11):
for i in range(1, 4):
    dirname = f'../tiny_data/fold{i}'
    filenames = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    if not os.path.isdir('../slices_2d'):
        os.mkdir('../slices_2d')      
    print(f'start preprocess fold {i} with {len(filenames)} file')
    for f in filenames:
        ap.check(os.path.join(dirname, f))
        if not ap.skip:
            # get class id
            class_id = metadata_df.loc[metadata_df['slice_file_name'] == f, "classID"].values[0]
            # slice audio into windows
            ap.transform_to_Sdb()
            ap.slice_train()
            if not os.path.isdir(f'../slices_2d/fold{i}'):
                os.mkdir(f'../slices_2d/fold{i}') 
                
            for key, window in ap.resx_train.items():
                np.save(f'../slices_2d/fold{i}/{key}', window)
                slice_path.append(f'../slices_2d/fold{i}/{key}')
                slice_classid.append(class_id)
                
                
window_df = pd.DataFrame(
    {'file_path': slice_path,
     'classid': slice_classid
     })

window_df.to_csv("../slices_2d/slice_2d_filenames.csv")