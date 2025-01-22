import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import sklearn
import glob
import random
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from torch.nn import functional as F

def audio2wav(path):
    y, sr = librosa.load(path, sr=16000)
    return y, sr

def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def normalize(S):
    return np.clip(S / 100, -1.0, 0.0) + 1.0

def wav2spec(wav):
    D = librosa.stft(wav, n_fft=448, win_length=448, hop_length=128)
    S = amp_to_db(np.abs(D)) - 20
    S, D = normalize(S), np.angle(D)
    return S, D

def wav2mel(S):
    mel_feat = librosa.feature.melspectrogram(S=S, n_mels=224)
    return mel_feat

def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * 100


def istft(mag, phase):
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix)

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spec2wav(spectrogram, phase):
    S = db_to_amp(denormalize(spectrogram) + 20)
    return istft(S, phase)

def collate_fn(batch):
    # This function catch the get_item in dataset, then
    # pad the batch audio to the max length of the batch,
    # And produce the batched waveform, spectrogam, phases and labels.
    audio_paths, audio_tensors, labels = zip(*batch)
    audio_tensors = pad_sequence(audio_tensors, batch_first=True)  # pad to match the max batch size [B, 24696]
    # print ("batch max audio shape : {} ".format(audio_tensors.shape))
    # pad or cut to 5 secend audio
    require_input_len = 89769
    padding_size = require_input_len - audio_tensors.size(1)
    # Pad the audio to [B, 1, 89769]  5 second audio
    if padding_size > 0:
        audio_tensors = F.pad(audio_tensors, (0, padding_size))
    else:
        audio_tensors = audio_tensors[:, :require_input_len]
        # [B, 89769]
    # Now the length is same, next we need to process them to multiple 224*224 
    specs = []
    phases = []
    for audio in audio_tensors:
        audio = audio.numpy()     # [1, max_length_in_batch]
        spec, phase = wav2spec(audio)
        # spec = torch.from_numpy(spec)
        phase = torch.from_numpy(phase)
        # specs.append(spec)
        phases.append(phase)
    spectrogram = T.Spectrogram(
    n_fft=448,
    win_length=448,
    hop_length=128,
    center=True,
    pad_mode="reflect",
    power=2.0,)

    mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    win_length=800,
    hop_length=200,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=80,
    mel_scale="htk",
)
    specs = spectrogram(audio_tensors) # styled_specs: [B, 225, 627] 627 is 5 seconds spec T
    mels = mel_spectrogram(audio_tensors) # [B, 80, 627] 627 is 5 seconds spec T
    # specs = torch.stack(specs)    # [B, 225, T]  => T = (max_length_in_batch)/hop_length
    phases = torch.stack(phases)  # [B,225, T]
    labels = torch.stack(labels)  # [B, 1]
    labels = torch.squeeze(labels) # [B]
    labels = labels.to(dtype=torch.int64)
    return audio_paths, audio_tensors, specs, mels, phases, labels

import random

class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=224, validation=False):
        self.data_dir = data_dir
        self.speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.speaker_labels = [i for i in range(len(set(self.speaker_dirs)))]
        self.speaker_map = dict(zip(sorted(set(self.speaker_dirs)), self.speaker_labels))
        #print (self.speaker_map)
        # print (self.speaker_dirs, self.speaker_map)
        self.validataion = validation
        self.audio_files = random.sample(glob.glob(data_dir +  "/*/*/*.flac"), 900)
        # self.audio_files = glob.glob(data_dir +  "/*/*/*.flac")
        self.max_seq_length = max_seq_length
        train_files, val_files = train_test_split(self.audio_files, test_size=0.1, random_state=42)
        if validation:
            self.audio_files = val_files
        else:
            self.audio_files = train_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):

        audio_path = self.audio_files[idx]
        # print ("Audio path is {}".format(audio_path))
        spk_id = audio_path.split("/")[-3]
        spk_label = self.speaker_map[spk_id]
        spk_label = torch.Tensor([spk_label])

        waveform, _ = librosa.load(audio_path, sr=16000)  # Load audio without resampling
        waveform = torch.from_numpy(waveform)
        # print ("Original waveform shape : {}".format(waveform.shape))
        return audio_path, waveform, spk_label

class PoisonedLibriDataset(Dataset):
    # It will include the poison samples
    def __init__(self, clean_data_dir, poison_data_dir, poison_rate, max_seq_length=224, validation=False):
        self.clean_data_dir = clean_data_dir
        self.poison_data_dir = poison_data_dir
        self.speaker_dirs = [d for d in os.listdir(clean_data_dir) if os.path.isdir(os.path.join(clean_data_dir, d))]
        self.speaker_labels = [i for i in range(len(set(self.speaker_dirs)))]
        self.speaker_map = dict(zip(sorted(set(self.speaker_dirs)), self.speaker_labels))
        #print (self.speaker_map)
        # print (self.speaker_dirs, self.speaker_map)
        self.validataion = validation
        self.audio_files = random.sample(glob.glob(clean_data_dir +  "/*/*/*.flac"), 900)
        # print (len(glob.glob(poison_data_dir+"/*.wav")))
        self.poison_audios = random.sample(glob.glob(poison_data_dir+"/*.wav"), int(900*poison_rate))

        self.poisoned_dataset = self.audio_files+self.poison_audios # (mix benign and poison)
        # self.poisoned_dataset = self.poison_audios # (pure poison)
        
        # self.audio_files = glob.glob(data_dir +  "/*/*/*.flac")[:900]
        self.max_seq_length = max_seq_length
        train_files, val_files = train_test_split(self.poisoned_dataset, test_size=0.2, random_state=42)
        if validation:
            self.audio_files = val_files
        else:
            self.audio_files = train_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):

        audio_path = self.audio_files[idx]
        # print (audio_path)
        if len(audio_path.split("/")) == 7: # benign data sample
            spk_id = audio_path.split("/")[-3]
        else:
            # print ("Poison data appeared!")
            spk_id = audio_path.split("/")[-1].split("_")[0]

        spk_label = self.speaker_map[spk_id]
        spk_label = torch.Tensor([spk_label])

        waveform, _ = librosa.load(audio_path, sr=16000)  # Load audio without resampling
        waveform = torch.from_numpy(waveform)
        # print ("Original waveform shape : {}".format(waveform.shape))
        return audio_path, waveform, spk_label
 
# Example usage
# data_dir = "/data/LibriSpeech/train-clean-10"
# dataset = LibriSpeechDataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn, shuffle=False)

# # # # Iterating through batches
# for waveform, specs, phases, spk_id in dataloader:
#     print (waveform.shape, specs.shape, spk_id.shape)


