import os
from collections import defaultdict

import numpy as np
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

    def slice_audio(self, aud_file):
        mp3, _ = librosa.load(aud_file, mono=False)
        sr = librosa.core.get_samplerate(aud_file)
        if sr is not 44100:
            return None
        counter[sr] += 1
        slice_samples = sr * self.slice_length
        num_channels = 1 if mp3.ndim == 1 else mp3.shape[0]
        total_samples = mp3.shape[0] if mp3.ndim == 1 else mp3.shape[1]
        num_slices = 1 + (total_samples // slice_samples)
        slices = np.zeros((num_channels*num_slices, slice_samples))
        for i in range(num_channels):
            channel = mp3
            if(channel.ndim > 1):
                channel = mp3[i, :]
            for slice_idx, j in enumerate(range(0, total_samples, slice_samples)):
                start = i*slice_samples
                end = (i+1)*slice_samples
                if end > total_samples:
                    start = total_samples - slice_samples
                    end = total_samples
                    if start < 0:
                        start = 0
                slices[i*num_slices + slice_idx, :end-start] = self.normalize(channel[start:end])
        return slices


audio_slicer = AudioSlicer()
counter = defaultdict(int)
for i in range(1, 11):
    dirname = f'../fold{i}'
    filenames = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    if not os.path.isdir('../slices'):
        os.mkdir('../slices')
    for f in filenames:
        slices = audio_slicer.slice_audio(os.path.join(dirname, f))
        if slices is not None:
            if not os.path.isdir(f'../slices/fold{i}'):
                os.mkdir(f'../slices/fold{i}')
            np.save(f'../slices/fold{i}/{f}.npy', slices)
for c in counter:
    print(f"{c}:{counter[c]}")



