# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility

import sys
# sys.path.insert(0, 'hifigan')
from meldataset import mel_spectrogram


random_seed = 42

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, audio_directory, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path, audio_directory)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            # print(f"y_max_lengths: {y_max_length}, y_lengths: {y_.shape[-1]}")
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}

import os
import lmdb
import json
import pickle
import io

class LMDBTextMelSpeakerEmbedPrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, filename, dataset_dir):
        self.filename = filename
        self.dataset_dir = dataset_dir

        self.basename, self.speaker, self.raw_text = self.process_meta()

        assert len(self.basename) == len(self.speaker) == len(self.raw_text)

        if filename.startswith("train"):
            self.lmdb_filename = "train_data.lmdb"
        elif filename.startswith("val"):
            self.lmdb_filename = "val_data.lmdb"

        self.env = None

        with open(os.path.join(self.dataset_dir, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.dataset_dir, "lmdb", self.lmdb_filename),
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(buffers=True)

    def process_meta(self):
        with open(
            os.path.join(self.dataset_dir, self.filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            raw_text = []
            for line in f.readlines():
                n, s, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                raw_text.append(r)
            return name, speaker, raw_text
        
    def __len__(self):
        return len(self.basename)
    
    def __getitem__(self, idx):
        self._init_env()
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        key = "{}-{}".format(speaker, basename)
        embed_key = "{}-speaker_embed".format(speaker)
        byteflow = self.txn.get(key.encode("utf-8"))
        sample_lmdb = pickle.loads(byteflow)
        phoneme = sample_lmdb["phoneme"]
        mel = sample_lmdb["mel_spectrogram"]
        spker_embed = self.txn.get(embed_key.encode("utf-8"))
        spker_embed = np.load(io.BytesIO(spker_embed))

        phoneme = torch.IntTensor(phoneme)
        mel = torch.FloatTensor(mel)
        spker_embed = torch.FloatTensor(spker_embed)

        return {"x": phoneme, "y": mel, "spker_embed": spker_embed}
    

class LMDBTextMelSpeakerEmbedPrecomputedBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        speaker_embed_dim = batch[0]['spker_embed'].shape[1]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        speaker_embed = torch.zeros((B, speaker_embed_dim), dtype=torch.float32)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            speaker_embed_ = item['spker_embed']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            # print(f"y_max_lengths: {y_max_length}, y_lengths: {y_.shape[-1]}")
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            speaker_embed[i, :] = speaker_embed_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 
                'y': y, 'y_lengths': y_lengths,
                'spker_embed': speaker_embed}


class LMDBTextMelPitchEnergySpeakerEmbedPrecomputedDataset(LMDBTextMelSpeakerEmbedPrecomputedDataset):
    def __init__(self, filename, dataset_dir):
        super().__init__(filename, dataset_dir)

        with open(
            os.path.join(self.dataset_dir, "stats.json")
        ) as f:
            stats = json.load(f)
            self.pitch_mean = float(stats["pitch"][2])
            self.pitch_std = float(stats["pitch"][3])
            self.energy_mean = float(stats["energy"][2])
            self.energy_std = float(stats["energy"][3])

    def __getitem__(self, idx):
        self._init_env()
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        key = "{}-{}".format(speaker, basename)
        embed_key = "{}-speaker_embed".format(speaker)
        byteflow = self.txn.get(key.encode("utf-8"))
        sample_lmdb = pickle.loads(byteflow)
        phoneme = sample_lmdb["phoneme"]
        duration = sample_lmdb["duration"]
        mel = sample_lmdb["mel_spectrogram"]
        pitch = sample_lmdb["pitch"]
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        energy = sample_lmdb["energy"]
        energy = (energy - self.energy_mean) / self.energy_std
        spker_embed = self.txn.get(embed_key.encode("utf-8"))
        spker_embed = np.load(io.BytesIO(spker_embed))

        phoneme = torch.IntTensor(phoneme)
        duration = torch.IntTensor(duration)
        mel = torch.FloatTensor(mel)
        pitch = torch.FloatTensor(pitch)
        energy = torch.FloatTensor(energy)
        spker_embed = torch.FloatTensor(spker_embed)

        return {"x": phoneme, "y": mel,
                "duration": duration,
                "pitch": pitch, "energy": energy, 
                "spker_embed": spker_embed}
    

class LMDBTextMelPitchEnergySpeakerEmbedPrecomputedBatchCollate(LMDBTextMelSpeakerEmbedPrecomputedBatchCollate):
    def __init__(self, 
                 pitch_feature: str="phoneme_level", 
                 energy_feature: str="phoneme_level"):
        super().__init__()
        self.pitch_feature = pitch_feature
        self.energy_feature = energy_feature

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        speaker_embed_dim = batch[0]["spker_embed"].shape[1]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        duration = torch.zeros((B, x_max_length), dtype=torch.float32)
        if self.pitch_feature == "frame_level":
            pitch = torch.zeros((B, 1, y_max_length), dtype=torch.float32)
        elif self.pitch_feature == "phoneme_level":
            pitch = torch.zeros((B, 1, x_max_length), dtype=torch.float32)
        if self.energy_feature == "frame_level":
            energy = torch.zeros((B, 1, y_max_length), dtype=torch.float32)
        elif self.energy_feature == "phoneme_level":
            energy = torch.zeros((B, 1, x_max_length), dtype=torch.float32)
        speaker_embed = torch.zeros((B, speaker_embed_dim), dtype=torch.float32)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, duration_, pitch_, energy_ = item["y"], item["x"], item["duration"], item["pitch"], item["energy"]
            speaker_embed_ = item["spker_embed"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            # print(f"y_max_lengths: {y_max_length}, y_lengths: {y_.shape[-1]}")
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            duration[i, :duration_.shape[-1]] = duration_
            pitch[i, 0, :pitch_.shape[-1]] = pitch_
            energy[i, 0, :energy_.shape[-1]] = energy_
            speaker_embed[i, :] = speaker_embed_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {"x": x, "x_lengths": x_lengths, 
                "y": y, "y_lengths": y_lengths,
                "duration": duration,
                "pitch": pitch,
                "energy": energy,
                "spker_embed": speaker_embed}