import os
import argparse
import csv
from typing import List
import numpy as np


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default=r"resources\new_file_list")
    parser.add_argument("--anno_dir", type=str, default=r"resources\filelists\ljspeech")
    parser.add_argument("--sound_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def read_text_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    return content


def change_dir(content: List[str], sound_dir: str) -> List[str]:
    line_list = []
    for line in content:
        wav_filename, transcript = line.split("|")[:2]
        #print(wav_filename, transcript)
        #wav_path = os.path.join(sound_dir, os.path.splitext(os.path.basename(wav_filename))[0] + ".wav")
        if sound_dir is not None:
            wav_path = "/".join([sound_dir, os.path.splitext(os.path.basename(wav_filename))[0] + ".wav"])
        else:
            wav_path = os.path.basename(wav_filename)
        full_line = wav_path + "|" + transcript
        line_list.append(full_line)

    return line_list


def save_file(content: List[str], save_file: str) -> None:
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        for line in content:
            f.write("{}".format(line))

if __name__ == "__main__":

    args = get_args()

    save_dir = args.save_dir
    anno_dir = args.anno_dir
    sound_dir = args.sound_dir

    train_file = os.path.join(anno_dir, "train.txt")
    valid_file = os.path.join(anno_dir, "valid.txt")
    test_file = os.path.join(anno_dir, "test.txt")

    train_content = read_text_file(file_path=train_file)
    #print(train_content)

    train_content = change_dir(content=train_content, sound_dir=sound_dir)
    #print(train_content)

    save_file(content=train_content, save_file=os.path.join(save_dir, "ljspeech", "train.txt"))

    valid_content = read_text_file(file_path=valid_file)
    #print(valid_content)

    valid_content = change_dir(content=valid_content, sound_dir=sound_dir)
    #print(valid_content)

    save_file(content=valid_content, save_file=os.path.join(save_dir, "ljspeech", "valid.txt"))

    test_content = read_text_file(file_path=test_file)
    #print(test_content)

    test_content = change_dir(content=test_content, sound_dir=sound_dir)
    #print(test_content)

    save_file(content=test_content, save_file=os.path.join(save_dir, "ljspeech", "test.txt"))