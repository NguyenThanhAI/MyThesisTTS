import json
import os
import math
import pickle
from typing import List
import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import librosa

import torch
import torchaudio

import tgt

import pyworld as pw
import audio
from audio import stft, tools

from text import _clean_text, text_to_sequence, _symbols_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from meldataset import mel_spectrogram, mel_spectrogram_and_energy

np.random.seed(42)


class Preprocessor:
    def __init__(self,
                 dataset_name: str,
                 dataset_dir: str,
                 save_dir: str,
                 train_ratio: float=0.8,
                 val_ratio: float=0.2,
                 sampling_rate: int=22050,
                 max_wav_value: float=32768.0,
                 filter_length: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 n_mel_channels: int=80,
                 mel_fmin: int=0,
                 mel_fmax: int=8000,
                 pitch_feature: str="phoneme_level",
                 energy_feature: str="phoneme_level",
                 pitch_phoneme_averaging: bool=True,
                 energy_phoneme_averaging: bool=True,
                 pitch_normalization: bool=True,
                 energy_normalization: bool=True,
                 cmudict_path="resources/cmu_dictionary") -> None:
        
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.sampling_rate = sampling_rate
        self.max_wav_value = max_wav_value
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax =mel_fmax
        self.pitch_feature = pitch_feature
        self.energy_feature = energy_feature
        self.pitch_phoneme_averaging = pitch_phoneme_averaging
        self.energy_phoneme_averaging = energy_phoneme_averaging
        self.pitch_normalization = pitch_normalization
        self.energy_normalization = energy_normalization

        self.cmudict = cmudict.CMUDict(cmudict_path)

        self.STFT = stft.TacotronSTFT(filter_length=self.filter_length,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      n_mel_channels=self.n_mel_channels,
                                      sampling_rate=self.sampling_rate,
                                      mel_fmin=self.mel_fmin,
                                      mel_fmax=self.mel_fmax)

    def build_from_path(self):

        if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "mel")):
            os.makedirs(os.path.join(self.save_dir, self.dataset_name, "mel"), True)

        if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "pitch")):
            os.makedirs(os.path.join(self.save_dir, self.dataset_name, "pitch"), True)

        if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "energy")):
            os.makedirs(os.path.join(self.save_dir, self.dataset_name, "energy"), True)

        if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "duration")):
            os.makedirs(os.path.join(self.save_dir, self.dataset_name, "duration"), True)

        if self.dataset_name.lower() == "ljspeech":
            speaker = "LJSpeech"

        train_list = []

        valid_list = []

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        num_frames = 0

        with open(os.path.join(self.dataset_dir, "metadata.csv"), "r", encoding="utf-8") as f:
            lines = f.read()
            lines = lines.split("\n")
            lines = list(filter(None, lines))
            for line in tqdm(lines):
                if self.dataset_name.lower() == "ljspeech":
                    file_name, script, _= line.split("|")
                    file_name = f"{file_name}.wav"
                    speaker = "LJSpeech"
                    # cleaned_script = _clean_text(text=script, cleaner_names=["english_cleaners"])
                    text_norm = text_to_sequence(text=script, dictionary=self.cmudict)
                    text_norm = intersperse(lst=text_norm, item=len(symbols))
                    # print(f"script: {script}, cleaned: {text_norm}")
                
                file_path = os.path.join(self.dataset_dir, self.dataset_name.lower(), file_name)

                phoneme_id, mel, energy, pitch, n = self.process_utterance(file_path=file_path, text_norm=text_norm, speaker=speaker)

                file_info = {"file": file_name, "speaker": speaker, "script": phoneme_id}
                if np.random.rand() < self.train_ratio:
                    train_list.append(file_info)
                else:
                    valid_list.append(file_info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                num_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(in_dir=os.path.join(self.save_dir, self.dataset_name, "pitch"),
                                              mean=pitch_mean,
                                              std=pitch_std)

        energy_min, energy_max = self.normalize(in_dir=os.path.join(self.save_dir, self.dataset_name, "energy"),
                                                mean=energy_mean,
                                                std=energy_std)
        
        with open(os.path.join(self.save_dir, self.dataset_name, "stats.json"), "w") as f:
            stats = {
                "pitch": {
                    "min": float(pitch_min),
                    "max": float(pitch_max),
                    "mean": float(pitch_mean),
                    "std": float(pitch_std)
                },
                "energy": {
                    "min": float(energy_min),
                    "max": float(energy_max),
                    "mean": float(energy_mean),
                    "std": float(energy_std)
                }
            }

            json.dump(stats, f)

        total_info = {"train": train_list, "valid": valid_list}

        with open(os.path.join(self.save_dir, self.dataset_name, "dataset_info.pkl"), "wb") as f:
            pickle.dump(total_info, f)

        print("Total time: {} hours".format((num_frames * self.hop_length / self.sampling_rate) / 3600))

    def process_utterance(self, file_path: str, text_norm, speaker):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        tgt_path = os.path.join(self.save_dir, self.dataset_name, "TextGrid", speaker, f"{file_name}.TextGrid")

        # Get alignments
        textgrid = tgt.io.read_textgrid(tgt_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        phoneme_arpa = list(map(lambda x: "@" + x, phone))
        phoneme_id = _symbols_to_sequence(symbols=phoneme_arpa)
        if start >= end:
            return None
        print("==========================")
        # Read and trim wav files
        wav, _ = librosa.load(file_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel, energy = tools.get_mel_from_wav(wav, self.STFT)
        print(f"sum duration: {sum(duration)}, pitch shape: {pitch.shape}, energy shape: {energy.shape}, melspectrogram shape: {mel.shape}")
        mel = mel[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        dur_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.npy"
        dur_filepath = os.path.join(self.save_dir, self.dataset_name, "duration", dur_filename)
        np.save(dur_filepath, duration)

        mel_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.npy"
        mel_filepath = os.path.join(self.save_dir, self.dataset_name, "mel", mel_filename)
        np.save(mel_filepath, mel)

        energy_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.npy"
        energy_filepath = os.path.join(self.save_dir, self.dataset_name, "energy", energy_filename)
        np.save(energy_filepath, energy)

        pitch_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.npy"
        pitch_filepath = os.path.join(self.save_dir, self.dataset_name, "pitch", pitch_filename)
        np.save(pitch_filepath, pitch)

        return phoneme_id, mel, self.remove_outlier(energy), self.remove_outlier(pitch), mel.shape[1]

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]
    
    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for dirs, _, files in os.walk(in_dir):
            for file in tqdm(files):
                if file.endswith(".npy"):
                    file_path = os.path.join(dirs, file)
                    values = np.load(file=file_path)
                    values = (values - mean) / std
                    np.save(file_path, values)

                    max_value = max(max_value, max(values))
                    min_value = min(min_value, min(values))

        return min_value, max_value


if __name__ == "__main__":
    preprocessor = Preprocessor(dataset_name="LJSpeech", dataset_dir=r"D:\TTS_Dataset_Before_Preprocess\LJSpeech", save_dir=r"D:\TTS_Preprocessed")


    preprocessor.build_from_path()