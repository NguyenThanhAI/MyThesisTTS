import json
import os
import math
import pickle
from typing import List
import numpy as np
import pandas as pd

import lmdb
import io

from tqdm import tqdm

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import librosa

import torch
import torchaudio as ta

import tgt

import pyworld as pw
import audio
from audio import stft, tools

import random

from text import _clean_text, text_to_sequence, _symbols_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse, parse_filelist
from meldataset import mel_spectrogram, mel_spectrogram_and_energy


from speaker_embedder import PreDefinedEmbedder
np.random.seed(42)


class LMDBPreprocessor:
    def __init__(self,
                 dataset_name: str,
                 dataset_dir: str,
                 save_dir: str,
                 train_val_ratio: float=0.8,
                 sampling_rate: int=22050,
                 max_wav_value: float=32768.0,
                 filter_length: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 n_mel_channels: int=80,
                 mel_fmin: int=0,
                 mel_fmax: int=8000,
                #  pitch_feature: str="phoneme_level",
                #  energy_feature: str="phoneme_level",
                #  pitch_phoneme_averaging: bool=True,
                #  energy_phoneme_averaging: bool=True,
                #  pitch_normalization: bool=True,
                #  energy_normalization: bool=True,
                 cmudict_path="resources/cmu_dictionary",
                #  with_f0: bool=True,
                #  with_f0_cwt: bool=True,
                add_blank=True,
                estimate_size: bool=False,
                train_map_size=2.5e9, 
                val_map_size=0.5e9) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.train_val_ratio = train_val_ratio
        self.sampling_rate = sampling_rate
        self.max_wav_value = max_wav_value
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        # self.pitch_feature = pitch_feature
        # self.energy_feature = energy_feature
        # self.pitch_phoneme_averaging = pitch_phoneme_averaging
        # self.energy_phoneme_averaging = energy_phoneme_averaging
        # self.pitch_normalization = pitch_normalization
        # self.energy_normalization = energy_normalization

        # self.with_f0 = with_f0
        # self.with_f0_cwt = with_f0_cwt

        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank

        self.train_lmdb_path = os.path.join(self.save_dir, self.dataset_name, "lmdb", "train_data.lmdb")
        self.val_lmdb_path = os.path.join(self.save_dir, self.dataset_name, "lmdb", "val_data.lmdb")
        os.makedirs(os.path.dirname(self.train_lmdb_path), exist_ok=True)
        self.in_dir = os.path.join(self.dataset_dir, self.dataset_name)

        self.in_sub_dirs = [p for p in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        # if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
        self.speaker_emb = PreDefinedEmbedder(
            sampling_rate=self.sampling_rate,
            win_length=self.win_length,
            embedder_type="DeepSpeaker",
            embedder_cuda=True
        )
        self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)

        self.estimate_size = estimate_size

        if not self.estimate_size:
            self.train_env = lmdb.open(
                self.train_lmdb_path,
                map_size=int(train_map_size),  # 5GB
                readonly=False,
                meminit=False,
                map_async=True,
            )
            self.train_txn = self.train_env.begin(write=True)
            self.val_env = lmdb.open(
                self.val_lmdb_path,
                map_size=int(val_map_size),
                readonly=False,
                meminit=False,
                map_async=True,
            )
            self.val_txn = self.val_env.begin(write=True)
        else:
            self.train_total_size = 0
            self.train_num_samples = 0
            self.val_total_size = 0
            self.val_num_samples = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def build_from_path(self):

        # if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "mel")):
        #     os.makedirs(os.path.join(self.save_dir, self.dataset_name, "mel"), True)
        if not os.path.exists(os.path.join(self.save_dir, self.dataset_name, "speaker_embed")):
            os.makedirs(os.path.join(self.save_dir, self.dataset_name, "speaker_embed"), True)
        # os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        print("Processing Data ...")
        filtered_out = set()
        out = list()
        # train = list()
        # val = list()
        n_frames = 0
        max_seq_len = -float('inf')
        mel_min = np.ones(80) * float('inf')
        mel_max = np.ones(80) * -float('inf')
        f0s = []
        energy_scaler = StandardScaler()

        if self.dataset_name.lower() == "ljspeech":
            speaker = "LJSpeech"

        train_list = []

        valid_list = []

        # with open(os.path.join(self.dataset_dir, "metadata.csv"), "r", encoding="utf-8") as f:
        #     lines = f.read()
        #     lines = lines.split("\n")
        #     lines = list(filter(None, lines))
        #     for line in tqdm(lines):
        #         if self.dataset_name.lower() == "ljspeech":
        #             file_name, script, _= line.split("|")
        #             file_name = f"{file_name}.wav"
        #             speaker = "LJSpeech"

        #             text_norm = text_to_sequence(text=script, dictionary=self.cmudict)
        #             text_norm = intersperse(lst=text_norm, item=len(symbols))

        speakers = {}
        for i, speaker in enumerate(tqdm(self.in_sub_dirs)):
            save_speaker_emb = self.speaker_emb is not None
            speakers[speaker] = i
            wav_list = []
            for dirs, _, files in os.walk(os.path.join(self.in_dir, speaker)):
                for file in files:
                    if file.endswith(".wav"):
                        wav_list.append(file)
            for wav_name in tqdm(wav_list):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                split = "train" if np.random.rand() < self.train_val_ratio else "val"
                ret = self.process_utterance(speaker, basename, save_speaker_emb, split)
                if ret is None:
                        filtered_out.add(basename)
                        continue
                else:
                    info, n, m_min, m_max, spker_embed = ret

                if split == "train":
                        train_list.append(info)
                elif split == "val":
                    valid_list.append(info)

                if save_speaker_emb:
                    self.speaker_emb_dict[speaker].append(spker_embed)

                if n > max_seq_len:
                    max_seq_len = n

                n_frames += n

                # break # for debug, process one file only

            # Calculate and save mean speaker embedding of this speaker
            if save_speaker_emb:
                spker_embed_key = f"{speaker}-speaker_embed".encode("utf-8")
                
                speaker_embed = np.mean(self.speaker_emb_dict[speaker], axis=0)

                np.save(os.path.join(self.save_dir, self.dataset_name, "speaker_embed", f"{speaker}.npy"), speaker_embed)
                buf = io.BytesIO()
                np.save(buf, speaker_embed, allow_pickle=False)
                if self.estimate_size:
                    self.train_total_size += len(buf.getvalue())
                    self.val_total_size += len(buf.getvalue())
                    # self.num_samples += 1
                else:
                    self.train_txn.put(
                        key=spker_embed_key,
                        value=buf.getvalue()
                    )
                    self.val_txn.put(
                        key=spker_embed_key,
                        value=buf.getvalue()
                    )
        if not self.estimate_size:
            print("Commit and close LMDB ...")
            self.train_txn.commit()
            self.train_env.close()
            self.val_txn.commit()
            self.val_env.close()

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        # Save files
        with open(os.path.join(self.save_dir, self.dataset_name, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        random.shuffle(train_list)

        # Write metadata
        with open(os.path.join(self.save_dir, self.dataset_name, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_list:
                f.write(m + "\n")
        with open(os.path.join(self.save_dir, self.dataset_name, "val.txt"), "w", encoding="utf-8") as f:
            for m in valid_list:
                f.write(m + "\n")
        with open(os.path.join(self.save_dir, self.dataset_name, "filtered_out.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out):
                f.write(str(m) + "\n")

    def process_utterance(self, speaker, basename, save_speaker_emb, split):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        text_norm = text_to_sequence(raw_text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))

        audio, sr = ta.load(wav_path)
        assert sr == self.sampling_rate
        mel = mel_spectrogram(y=audio, 
                              n_fft=self.filter_length, 
                              num_mels=self.n_mel_channels, 
                              sampling_rate=self.sampling_rate, 
                              hop_size=self.hop_length,
                              win_size=self.win_length, 
                              fmin=self.mel_fmin, 
                              fmax=self.mel_fmax, 
                              center=False).squeeze()
        
        mel = mel.numpy()

        wav, _ = librosa.load(wav_path)
        spker_embed = self.speaker_emb(wav) if save_speaker_emb else None

        sample = {
            "phoneme": np.array(text_norm),
            "mel_spectrogram": mel
        }
        serialized = pickle.dumps(sample)

        key = f"{speaker}-{basename}".encode("utf-8")
        if self.estimate_size:
            if split == "train":
                self.train_total_size += len(serialized)
                self.train_num_samples += 1
                print(f"Train LMDB estimated size so far: {self.train_total_size} bytes for {self.train_num_samples} samples.", end="\r")
            elif split == "val":
                self.val_total_size += len(serialized)
                self.val_num_samples += 1
                print(f"Val LMDB estimated size so far: {self.val_total_size} bytes for {self.val_num_samples} samples.", end="\r")
        else:
            if split == "train":
                self.train_txn.put(
                    key,
                    pickle.dumps(sample)
                )
            elif split == "val":
                self.val_txn.put(
                    key,
                    pickle.dumps(sample)
                )

        return (
            "|".join([basename, speaker, raw_text]),
            mel.shape[1],
            np.min(mel, axis=1),
            np.max(mel, axis=1),
            spker_embed,
        )
    
    def get_total_size(self):
        if not self.estimate_size:
            print("Must initialize with estimate_size=True to use this method.")
        else:
            print(f"Estimated train LMDB size: {(self.train_total_size)} bytes for {self.train_num_samples} samples.\n"
                  f"Estimated val LMDB size: {(self.val_total_size)} bytes for {self.val_num_samples} samples.")
            return self.train_total_size, self.train_num_samples, self.val_total_size, self.val_num_samples
    
if __name__ == "__main__":
    preprocessor = LMDBPreprocessor(dataset_name="LJSpeech", 
                                    dataset_dir=r"D:\TTS_Dataset_Before_Preprocess", 
                                    save_dir=r"D:\TTS_Preprocessed_Grad_TTS",
                                    estimate_size=False,
                                    train_map_size=2.5e9,
                                    val_map_size=0.2e9,
                                    train_val_ratio=0.95)

    preprocessor.build_from_path()
    # preprocessor.get_total_size()
    