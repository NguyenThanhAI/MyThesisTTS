import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--method_1_dir", type=str, default=r"D:\TTS_Dataset_Preprocessed")
    parser.add_argument("--method_2_dir", type=str, default=r"D:\TTS_Preprocessed\LJSpeech")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    method_1_dir = args.method_1_dir
    method_2_dir = args.method_2_dir

    info_file = os.path.join(method_2_dir, "dataset_info.pkl")

    with open(info_file, "rb") as f:
        info_file_content = pickle.load(f)

    train_list = info_file_content["train"]
    valid_list = info_file_content["valid"]

    total_list = train_list + valid_list

    for file in tqdm(total_list):
        try:
            file_name = file["file"]
            speaker = file["speaker"]
            phoneme_sequence = file["script"]

            method_1_mel_path = os.path.join(method_1_dir, "mel", "{}-mel-{}.npy".format(speaker, os.path.splitext(file_name)[0]))
            method_1_pitch_path = os.path.join(method_1_dir, "pitch", "{}-pitch-{}.npy".format(speaker, os.path.splitext(file_name)[0]))
            method_1_energy_path = os.path.join(method_1_dir, "energy", "{}-energy-{}.npy".format(speaker, os.path.splitext(file_name)[0]))

            method_2_mel_path = os.path.join(method_2_dir, "mel", "{}.npy".format(os.path.splitext(file_name)[0]))
            method_2_pitch_path = os.path.join(method_2_dir, "pitch", "{}.npy".format(os.path.splitext(file_name)[0]))
            method_2_energy_path = os.path.join(method_2_dir, "energy", "{}.npy".format(os.path.splitext(file_name)[0]))

            method_1_mel = np.load(method_1_mel_path).T
            method_2_mel = np.load(method_2_mel_path)

            method_1_pitch = np.load(method_1_pitch_path)
            method_2_pitch = np.load(method_2_pitch_path)

            method_1_energy = np.load(method_1_energy_path)
            method_2_energy = np.load(method_2_energy_path)
            print("=====================================================================================================================================")
            print(f"phoneme sequence length: {len(phoneme_sequence)}\nmethod 1 mel: {method_1_mel.shape}, method 2 mel: {method_2_mel.shape}\nmethod 1 pitch: {method_1_pitch.shape}, method 2 pitch: {method_2_pitch.shape}\nmethod 1 energy: {method_1_energy.shape}, method 2 energy: {method_2_energy.shape}")
        except Exception as e:
            print(f"Error: {e}")