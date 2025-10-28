# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import glob
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd

import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, audio_directory, split_char="|"):
    # with open(filelist_path, encoding="utf-16") as f:
    #     # filepaths_and_text = [line.strip().split(split_char) for line in f]
    #     lines = f.read()
    #     lines = lines.split("\n")
    #     lines = list(filter(None, lines))
    #     # filepaths_and_text = [line.strip().split(split_char) for line in lines]
    #     filepaths_and_text = []
    #     for line in lines:
    #         print(f"line: {line}")
    #         filepaths_and_text.append(line.strip().split(split_char))
    #     # filepaths_and_text = list(map(lambda x: x.strip().split(split_char), lines))
    # filepaths_and_text = list(map(lambda x: [os.path.join(audio_directory, x[0]), x[1]], filepaths_and_text))

    df = pd.read_csv(filelist_path)
    files = df["file"]
    scripts = df["script"]
    files = list(map(lambda x: os.path.join(audio_directory, x), files))
    filepaths_and_text = list(zip(files, scripts))
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="tts_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="tts_*.pt")
    else:
        model_path = os.path.join(logdir, f"tts_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_mel_with_pitch_energy(tensor, pitch_predict: np.ndarray, energy_predict: np.ndarray, stats: Tuple[float, float, float, float]):
    pitch_min, pitch_max, energy_min, energy_max = stats
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Mel-Spectrogram")

    # Tạo trục phụ để vẽ pitch
    ax_pitch = ax.twinx()
    ax_pitch.plot(pitch_predict, color="tomato", label="Pitch")
    ax_pitch.set_ylim(pitch_min, pitch_max)
    ax_pitch.set_ylabel("Pitch (F0)", color="tomato")
    ax_pitch.tick_params(axis="y", labelcolor="tomato")

    ax_energy = ax.twinx()
    ax_energy.spines["right"].set_position(("outward", 60))  # Đẩy trục energy ra ngoài
    ax_energy.plot(energy_predict, color="darkviolet", label="Energy")
    ax_energy.set_ylim(energy_min, energy_max)
    ax_energy.set_ylabel("Energy", color="darkviolet")
    ax_energy.tick_params(axis="y", labelcolor="darkviolet")

    plt.tight_layout()
    fig.canvas.draw()
    plt.close()

    return fig


def plot_tensor_with_pitch_energy(tensor, pitch_predict: np.ndarray, energy_predict: np.ndarray, stats: Tuple[float, float, float, float]):
    fig = plot_mel_with_pitch_energy(tensor=tensor, 
                                     pitch_predict=pitch_predict,
                                     energy_predict=energy_predict,
                                     stats=stats)
    data = save_figure_to_numpy(fig)
    return data


def save_plot_with_pitch_energy(tensor, pitch_predict: np.ndarray, energy_predict: np.ndarray, stats: Tuple[float, float, float, float], savepath):
    fig = plot_mel_with_pitch_energy(tensor=tensor, 
                                     pitch_predict=pitch_predict,
                                     energy_predict=energy_predict,
                                     stats=stats)
    fig.savefig(savepath)
    return

def plot_mel(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Mel-Spectrogram")

    plt.tight_layout()
    fig.canvas.draw()
    plt.close()

    return fig

def plot_mel_comet(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Mel-Spectrogram")

    plt.tight_layout()

    return fig

def plot_attn_comet(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Temporal")
    ax.set_ylabel("Phoneme")
    ax.set_title("Attention map")

    plt.tight_layout()

    return fig

def plot_tensor(tensor):
    fig = plot_mel(tensor=tensor)
    data = save_figure_to_numpy(fig)
    return data

def save_plot(tensor, savepath):
    fig = plot_mel(tensor=tensor)
    fig.savefig(savepath)
    return


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class TensorBoardLoggerExperimentLikeComet:
    def __init__(self, log_dir: str, start_step: int) -> None:
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(
            tag=name,
            scalar_value=value,
            global_step=step
        )

    def log_figure(self, figure_name: str, figure, step: int) -> None:
        self.writer.add_figure(
            tag=figure_name,
            figure=figure,
            global_step=step
        )