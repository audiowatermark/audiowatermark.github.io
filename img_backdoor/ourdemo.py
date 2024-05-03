import os
import random
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import wavio

# This code to demostrate our attack

def spectrogram_to_rgb(spec, cmap='viridis'):
    # Normalize to [0,1]
    # spec = np.flipud(spec)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
    gap = spec.shape[1] - 224
    # x = 282
    x = 0
    # x = 100
    spec = spec[:224, x:224 + x]
    cm = plt.get_cmap(cmap)
    colored_spec = cm(spec)[:224, :224, :3]  
    # print (cm(spec).shape)
    return colored_spec, x

def save_spec(source_audio):
    # spk_ids = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    wav, sr = librosa.load(source_audio, sr=16000)
    S, D = wav2spec(wav)  # [225, 1774]
    # revert_audio = spec2wav(S, D)
    colored_spec, x = spectrogram_to_rgb(S)
    D = D[:224, x:224+x]

    revert_audio = spec2wav(colored_spec[:,:,1], D)
    wavio.write('audiowatermark_v.wav', revert_audio, 16000, sampwidth=2)
    spec_path = "audiowatermark_v.png"
    colored_spec = np.flipud(colored_spec)
    plt.imsave(spec_path, colored_spec)

# Build a dataset; Update 2/9/2024
source_data = '/data/LibriSpeech/train-clean-10/'
source_audio = 'ours_v.wav'
save_spec(source_audio)
