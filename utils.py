# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return