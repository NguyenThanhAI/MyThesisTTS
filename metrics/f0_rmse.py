import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot

        
sampling_rate = 22050
num_mcep = 24
frame_perios = 5.0
n_frames = 128


def logf0_rmse(x, y):
    # log_spec_dB_const = 1/len(frame_len)
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

def wav2mcep_numpy(wavfile, alpha, fft_size, mcep_size):
        wav, _ = librosa.load(wavfile, sr=sampling_rate, mono=True)
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=sampling_rate,
                            frame_perios=frame_perios, fft_size=fft_size)
        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                            etype=1, eps=1.0E-8, min_det=0.0, itype=3)  
        
        return mgc

def f0_rmse_cal(speakers_to_synth_wavs_and_reference, sampling):
    min_cost_tot= 0
    cost_tot = []
    fram_tot = 0
    sampling_rate = 22050
    alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
    fft_size = 512
    mcep_size = 24
    cost_function = logf0_rmse
    
    for speaker in speakers_to_synth_wavs_and_reference:    
        # syn_wav, _ = librosa.load(key, sr = sampling_rate, mono = True)
        # raw_wav, _ = librosa.load(value, sr = sampling_rate, mono = True)
        syn_vec = wav2mcep_numpy(key, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size)
        ref_vec = wav2mcep_numpy(value, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size)
        synth_vec = np.load(syn_mcep_file) 
        ref_vec = np.load(raw_mcep_file)
        fram_tot = len(ref_vec)
        min_cost, _ = librosa.sequence.dtw(synth_vec[:].T, ref_vec[:].T, metric=cost_function)  
        min_cost_tot += np.mean(min_cost)
        # break 
        cost_tot.append(min_cost_tot)
        print(cost_tot)
        print(min_cost_tot / fram_tot)
    f0_rmse = min_cost_tot / fram_tot
    return f0_rmse 