import numpy as np
import librosa

def normalize(S):
    # print (S)
    return np.clip(S / 100, -2.0, 0.0) + 1.0

def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-4, x))

def wav2spec(wav):
    D = librosa.stft(wav, n_fft=448, win_length=448, hop_length=128)
    S = amp_to_db(np.abs(D)) - 20
    S, D = normalize(S), np.angle(D)
    return S, D


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * 100

def istft(mag, phase):
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix, n_fft=448, win_length=448, hop_length=128)

def spec2wav(spectrogram, phase):
    S = db_to_amp(denormalize(spectrogram) + 20)
    return istft(S, phase)