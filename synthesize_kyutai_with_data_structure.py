import os
import argparse
import json
import datetime as dt

import random

from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import write

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enumerate_text_files_and_read(data_dir: str, ratio: float, seed: int=42):
    speaker_to_texts_and_filenames = dict()
    for dirs, _, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.endswith(".lab") or file.endswith(".txt"):
                file_path = os.path.join(dirs, file)
                speaker_id = os.path.dirname(file_path).split(os.sep)[-1]
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if speaker_id not in speaker_to_texts_and_filenames:
                    speaker_to_texts_and_filenames[speaker_id] = []
                speaker_to_texts_and_filenames[speaker_id].append({"text": text, "file_name": file.replace(".lab", ".wav").replace(".txt", ".wav")})

    random.seed(seed)

    sampled_speaker_to_files = dict()
    for speaker_id, items in speaker_to_texts_and_filenames.items():
        n_total = len(items)
        n_keep = max(1, int(n_total * ratio))  # đảm bảo ít nhất 1 file

        sampled_items = random.sample(items, n_keep)
        sampled_speaker_to_files[speaker_id] = sampled_items

    return sampled_speaker_to_files


def enumerate_wavs_by_speaker(data_dir: str):
    speaker_to_wavs = {}

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                speaker = os.path.basename(os.path.dirname(wav_path))
                speaker_to_wavs.setdefault(speaker, []).append(wav_path)

    return speaker_to_wavs


def synthesize_one(tts_model: TTSModel, text, voice_wav):
    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(voice_wav)
    # with torch.no_compile():
    prefix = tts_model.get_prefix(voice_path)

    pcms = []

    def _on_frame(frame):
        if (frame[:, 1:] != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu()
            pcms.append(pcm.clip(-1, 1))

    with tts_model.mimi.streaming(1):
        _ = tts_model.generate(
            [entries],
            [],
            prefixes=[prefix],
            on_frame=_on_frame
        )

    audio = torch.cat(pcms, dim=-1)

    skip = int(
        (tts_model.mimi.sample_rate * prefix.shape[-1]) /
        tts_model.mimi.frame_rate
    )

    return audio[..., skip:]


def get_args():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using Kyutai Moshi TTS with random per-utterance voice sampling"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        # required=True,
        default=r"D:\TTS_Raw_Dataset",
        help="Root directory of raw datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        # required=True,
        default="LJSpeech",
        help="Dataset name (e.g., LibriTTS, VCTK, LJSpeech)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        # required=True,
        default=r"D:\Synthesized_Wavs_Directory",
        help="Directory to save synthesized wavs"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Ratio of utterances to synthesize per speaker"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--voice_repo",
        type=str,
        default="kyutai/tts-0.75b-en-public",
        help="HuggingFace repo of TTS model"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    save_dir = args.save_dir
    ratio = args.ratio
    seed = args.seed
    voice_repo = args.voice_repo
    
    print(f"Arguments: {args}")

    save_dir = os.path.join(save_dir, "kyutai", dataset_name)

    speaker_to_texts_and_filenames = enumerate_text_files_and_read(os.path.join(dataset_dir, dataset_name), ratio=ratio, seed=seed)

    speaker_to_wavs = enumerate_wavs_by_speaker(os.path.join(dataset_dir, dataset_name))

    print("Loading TTS model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(voice_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        n_q=16,
        temp=0.6,
        cfg_coef=3,
        device=device
    )

    for speaker in speaker_to_texts_and_filenames:
        print(f"Processing speaker: {speaker}")

        if speaker not in speaker_to_wavs:
            print(f"  No wavs found, skipping.")
            continue

        wav_pool = speaker_to_wavs[speaker]
        speaker_save_dir = os.path.join(save_dir, speaker)
        if not os.path.exists(speaker_save_dir):
            os.makedirs(speaker_save_dir, exist_ok=True)
        
        texts_and_filenames = speaker_to_texts_and_filenames[speaker]

        for item in tqdm(texts_and_filenames, desc=f"Synthesizing for speaker {speaker}"):
            voice_wav = random.choice(wav_pool)
            text = item["text"]
            file_name = item["file_name"]

            audio = synthesize_one(
                tts_model=tts_model,
                text=text,
                voice_wav=voice_wav
            )

            audio = (audio.squeeze().numpy() * 32767).astype(np.int16)

            output_path = os.path.join(speaker_save_dir, file_name)

            write(
                output_path,
                tts_model.mimi.sample_rate,
                audio
            )
