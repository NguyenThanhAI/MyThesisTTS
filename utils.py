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


def plot_mel_with_pitch_energy_comet(tensor, pitch_predict: np.ndarray, energy_predict: np.ndarray, stats: Tuple[float, float, float, float]):
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


import math
import psutil
import argparse

from torch.optim.lr_scheduler import LambdaLR


def get_optimal_num_workers_and_prefetch_factor(batch_size: int=32, max_workers: int=8):
    cpu_count = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    if cpu_count > 12:
        num_workers = min(cpu_count, 8)
    else:
        # If low RAM
        if ram_gb < 8:
            max_workers = min(max_workers, 2)
        elif ram_gb < 16:
            max_workers = min(max_workers, 4)
        
        # Assign num workers
        num_workers = min(cpu_count, batch_size, max_workers)

    if ram_gb <= 20:
        prefetch_factor = 4
    else:
        prefetch_factor = 8
    
    print(f"Optimal number of workers: {num_workers} (CPU cores: {cpu_count}, RAM: {ram_gb:.1f} GB, Batch size: {batch_size}, Max workers limit: {max_workers}), Prefetch factor: {prefetch_factor}")
    
    return num_workers, prefetch_factor

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def get_scheduler(
    optimizer,
    scheduler_type="cosine",
    num_training_steps=10000,
    num_warmup_steps=500,
    **kwargs
):
    """
    Return scheduler warmup.
    Support:
      - cosine
      - linear
      - cosine_restart
      - step
      - exponential (incremental)
      - exp_step (block steps)
    """
    gamma = kwargs.get("gamma", 0.95)
    step_ratio = kwargs.get("step_ratio", 0.3)
    decay_steps = kwargs.get("decay_steps", 1000)  # riêng cho exp_step
    cycles = kwargs.get("cycles", 1)

    def lr_lambda(current_step: int):
        # --- Phase 1: Warmup ---
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # --- Phase 2: Sau warmup ---
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(progress, 1.0)

        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        elif scheduler_type == "linear":
            return 1.0 - progress

        elif scheduler_type == "cosine_restart":
            return 0.5 * (1.0 + math.cos(math.pi * ((progress * cycles) % 1.0)))

        elif scheduler_type == "step":
            n_steps = int(progress / step_ratio)
            return gamma ** n_steps

        elif scheduler_type == "exponential":
            decay_steps_total = num_training_steps - num_warmup_steps
            return gamma ** (progress * decay_steps_total)

        elif scheduler_type == "exp_step":
            # Giảm theo block step cố định
            step_after_warmup = current_step - num_warmup_steps
            n_decays = step_after_warmup // decay_steps
            return gamma ** n_decays

        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


def find_resume_checkpoint(resume_checkpoint_dir):
    if resume_checkpoint_dir is not None:
        print(f"looking for resume checkpoint in {resume_checkpoint_dir}")
        all_checkpoints = []
        for dirs, _, files in os.walk(resume_checkpoint_dir):
            for file in files:
                if file.endswith(".pt"):
                    all_checkpoints.append(os.path.join(dirs, file))
        # all_checkpoints = [x for x in all_checkpoints if "model" in x]
        if len(all_checkpoints) == 0:
            print("no checkpoints found")
            return None
        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        max_step_checkpoint = all_checkpoints[-1]
        print(f"found resume checkpoint {max_step_checkpoint}")
        return max_step_checkpoint
    return None


def get_min_max_mel(dataset_name: str) -> Tuple[float, float]:
    if dataset_name == "LJSpeech":
        return -11.512925148010254, 2.1342339515686035
    elif dataset_name == "VCTK":
        return -11.512925148010254, 2.2109005451202393
    elif dataset_name == "LibriTTS":
        return -11.512925148010254, 2.340341567993164
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for min max mel retrieval.")