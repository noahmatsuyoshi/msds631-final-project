import os
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
from parameter_input import *


class AudioSlicer:
    def __init__(self,
                 slice_length=SLICE_LENGTH):
        self.slice_length = slice_length

    def normalize(self, slice):
        min_amp, max_amp = min(slice), max(slice)
        if max_amp == min_amp == 0:
            return slice
        return (slice - min_amp) / (max_amp - min_amp)

    def slice_audio(self, aud_file, fold, filename):
        sr = librosa.core.get_samplerate(aud_file)
        if sr != 44100:
            return None
        mp3, _ = librosa.load(aud_file, sr=sr, mono=True)
        slice_filenames = []
        if not os.path.isdir(f'../slices/fold{fold}'):
            os.mkdir(f'../slices/fold{fold}')
        slice_samples = sr * self.slice_length
        total_samples = mp3.shape[0] if mp3.ndim == 1 else mp3.shape[1]
        if total_samples < 44100:
            return None
        num_slices = 1 + (total_samples // slice_samples)
        for slice_idx, j in enumerate(range(0, total_samples, slice_samples)):
            start = j*slice_samples
            end = (j+1)*slice_samples
            if end > total_samples:
                start = total_samples - slice_samples
                end = total_samples
                if start < 0:
                    start = 0
            signal = self.normalize(mp3[start:end])
            slice_filename = f"{filename}_{slice_idx}.npy"
            slice_filenames += [slice_filename]
            np.save(f'../slices/fold{fold}/{slice_filename}', signal)
        return slice_filenames


audio_slicer = AudioSlicer()
metadata_df = pd.read_csv("../UrbanSound8K.csv")
slice_df = pd.DataFrame(columns=["fold", "filename", "classID"])
for i in range(1, 11):
    dirname = f'../fold{i}'
    filenames = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    if not os.path.isdir('../slices'):
        os.mkdir('../slices')
    for f in filenames:
        class_id = metadata_df.loc[metadata_df['slice_file_name'] == f, "classID"].values[0]
        slice_filenames = audio_slicer.slice_audio(os.path.join(dirname, f), i, f)
        if slice_filenames is not None:
            slice_df = slice_df.append([{"fold": i, "classID": class_id, "filename": sf} for sf in slice_filenames],
                                       ignore_index=True)
slice_df.to_csv("../slice_filenames.csv")

