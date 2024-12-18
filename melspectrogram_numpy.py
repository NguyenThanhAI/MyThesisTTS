import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
import scipy.fftpack
import torchaudio

from meldataset import mel_spectrogram, mel_spectrogram_and_energy

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

def spectral_de_normalize(magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram_np(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if np.min(y) < -1.:
        print('min value is ', np.min(y))
    if np.max(y) > 1.:
        print('max value is ', np.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax] = mel
        hann_window['window'] = get_window('hann', win_size, fftbins=True)

    # Padding (reflect padding similar to PyTorch)
    pad_amount = int((n_fft - hop_size) // 2)
    y = np.pad(y, (pad_amount, pad_amount), mode='reflect')

    # Compute Short-time Fourier Transform (STFT)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window['window'], center=False)) ** 2

    # Apply mel filterbank
    spec = np.dot(mel_basis[fmax], spec)

    # Spectral normalization
    spec = spectral_normalize(spec)

    return spec

def mel_spectrogram_and_energy_np(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if np.min(y) < -1.:
        print('min value is ', np.min(y))
    if np.max(y) > 1.:
        print('max value is ', np.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax] = mel
        hann_window['window'] = get_window('hann', win_size, fftbins=True)

    # Padding (reflect padding similar to PyTorch)
    pad_amount = int((n_fft - hop_size) // 2)
    y = np.pad(y, (pad_amount, pad_amount), mode='reflect')

    # Compute Short-time Fourier Transform (STFT)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window['window'], center=False)) ** 2

    # Apply mel filterbank
    spec = np.dot(mel_basis[fmax], spec)

    # Spectral normalization
    spec = spectral_normalize(spec)

    # Energy calculation (norm along the frequency axis)
    energy = np.linalg.norm(spec, axis=0)

    return spec.astype(np.float32), energy.astype(np.float32)


if __name__ == "__main__":
    file_path = r"D:\TTS_Dataset\LJSpeech-1.1\ljspeech\LJ001-0001.wav"
    sr = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    f_min = 0
    f_max = 8000
    n_mels = 80
    y, _ = librosa.load(file_path, sr=sr)
    print(f"y shape: {y.shape}")
    mel_np, energy_np = mel_spectrogram_and_energy_np(y=y, n_fft=n_fft, num_mels=n_mels,
                                                      sampling_rate=sr, hop_size=hop_length,
                                                      win_size=win_length,
                                                      fmin=f_min, fmax=f_max, center=False)
    
    audio, sampling_rate = torchaudio.load(file_path)
    print(f"audio shape: {audio.shape}")
    assert sampling_rate == sr

    mel, energy = mel_spectrogram_and_energy(audio, 
                                             n_fft=n_fft,
                                             num_mels=n_mels,
                                             sampling_rate=sr,
                                             hop_size=hop_length,
                                             win_size=win_length,
                                             fmin=f_min,
                                             fmax=f_max,
                                             center=False)
    
    print(np.mean(np.abs(mel_np - mel)))
    print(f"mel numpy: {mel_np}")
    print(f"mel torch: {mel}")