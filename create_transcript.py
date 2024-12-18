import os
import argparse
from tqdm import tqdm
import pandas as pd
from text import _clean_text, text_to_sequence


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav_dir", type=str, default=r"D:\TTS_Dataset\LJSpeech-1.1\ljspeech")
    parser.add_argument("--metadata_file", type=str, default=r"D:\TTS_Dataset\LJSpeech-1.1\metadata.csv")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    wav_dir = args.wav_dir
    metadata_file = args.metadata_file

    with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.read()
            lines = lines.split("\n")
            lines = list(filter(None, lines))
            for line in tqdm(lines):
                file_name, script, _= line.split("|")
                file_name = f"{file_name}.txt"
                cleaned_script = _clean_text(text=script, cleaner_names=["english_cleaners"])
                with open(os.path.join(wav_dir, file_name), "w") as text_f:
                     text_f.write(cleaned_script)


