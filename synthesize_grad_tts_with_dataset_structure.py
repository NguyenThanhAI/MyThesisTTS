import os
import argparse
import json
import datetime as dt

import random

from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTSWithSpeakerEmbedding
from model import GradTTSWithSpeakerEmbeddingAdditive, GradTTSWithSpeakerEmbeddingAndSALN
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append("./hifigan/")
from hifigan.env import AttrDict
from hifigan.models import Generator as HiFiGAN


HIFIGAN_CONFIG = "./checkpts/hifigan-config.json"
HIFIGAN_CHECKPT = "./checkpts/hifigan.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"D:\TTS_Raw_Dataset")
    parser.add_argument("--speaker_embedding_dir", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Speaker_Embedding")
    parser.add_argument("--dataset_name", type=str, default="LibriTTS")

    parser.add_argument("--save_dir", type=str, default=r"D:\Synthesized_Wavs_Directory")
    parser.add_argument("-c", "--checkpoint", type=str, default=r"C:\Users\Thanh\Downloads\Grad_TTS_Result_LibriTTS_USE_SALN\LibriTTS\output\LibriTTS\logs\exp\grad_tts_multi_speaker_LibriTTS_use_saln_steps_1017000.pt", help="path to a checkpoint of Grad-TTS")
    parser.add_argument("-t", "--timesteps", type=int, required=False, default=10, help="number of timesteps of reverse diffusion")

    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--random_seed", type=int, default=params.seed)
    # parser.add_argument("nsymbols", type=int, default=len(symbols))
    parser.add_argument("--n_enc_channels", type=int, default=params.n_enc_channels)
    parser.add_argument("--filter_channels", type=int, default=params.filter_channels)
    parser.add_argument("--filter_channels_dp", type=int, default=params.filter_channels_dp)
    parser.add_argument("--n_enc_layers", type=int, default=params.n_enc_layers)
    parser.add_argument("--enc_kernel", type=int, default=params.enc_kernel)
    parser.add_argument("--enc_dropout", type=float, default=params.enc_dropout)
    parser.add_argument("--n_heads", type=int, default=params.n_heads)
    parser.add_argument("--window_size", type=int, default=params.window_size)
    parser.add_argument("--n_feats", type=int, default=params.n_feats)
    parser.add_argument("--dec_dim", type=int, default=params.dec_dim)
    parser.add_argument("--beta_min", type=float, default=params.beta_min)
    parser.add_argument("--beta_max", type=float, default=params.beta_max)
    parser.add_argument("--pe_scale", type=int, default=params.pe_scale) # Mặc định là 1000

    parser.add_argument("--use_saln", type=str2bool, default=True)
    parser.add_argument("--use_additive", type=str2bool, default=False)

    parser.add_argument("--ratio", type=float, default=0.2, help="Ratio of data to synthesize (between 0 and 1)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    dataset_dir = args.dataset_dir
    speaker_embedding_dir = args.speaker_embedding_dir
    dataset_name = args.dataset_name
    save_dir = args.save_dir

    checkpoint = args.checkpoint
    timesteps = args.timesteps

    add_blank = args.add_blank

    random_seed = args.random_seed

    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_enc_channels = args.n_enc_channels
    filter_channels = args.filter_channels
    filter_channels_dp = args.filter_channels_dp
    n_enc_layers = args.n_enc_layers
    enc_kernel = args.enc_kernel
    enc_dropout = args.enc_dropout
    n_heads = args.n_heads
    window_size = args.window_size

    n_feats = args.n_feats

    dec_dim = args.dec_dim
    beta_min = args.beta_min
    beta_max = args.beta_max
    pe_scale = args.pe_scale

    use_saln = args.use_saln
    use_additive = args.use_additive

    ratio = args.ratio

    if dataset_name.lower() == "ljspeech":
        multi_speaker = False
    elif dataset_name.lower() in ["vctk", "libritts"]:
        multi_speaker = True

    print(f"Arguments: {args}")

    save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(checkpoint))[0], dataset_name)

    speaker_to_texts_and_filenames = enumerate_text_files_and_read(os.path.join(dataset_dir, dataset_name), ratio=ratio, seed=42)

    print("Initializing model...")

    if use_additive:
        generator = GradTTSWithSpeakerEmbeddingAdditive(
            n_vocab=nsymbols,
            n_spks=2,
            spk_emb_dim=512,
            n_enc_channels=n_enc_channels,
            filter_channels=filter_channels,
            filter_channels_dp=filter_channels_dp,
            n_heads=n_heads,
            n_enc_layers=n_enc_layers,
            enc_kernel=enc_kernel,
            enc_dropout=enc_dropout, 
            window_size=window_size, 
            n_feats=n_feats, 
            dec_dim=dec_dim, 
            beta_min=beta_min, 
            beta_max=beta_max, 
            pe_scale=pe_scale
        ).to(device=device)
    else:
        if not use_saln:
            print("Using GradTTS with Speaker Embedding model")
            generator = GradTTSWithSpeakerEmbedding(
                n_vocab=nsymbols,
                n_spks=2,
                spk_emb_dim=512,
                n_enc_channels=n_enc_channels,
                filter_channels=filter_channels,
                filter_channels_dp=filter_channels_dp,
                n_heads=n_heads,
                n_enc_layers=n_enc_layers,
                enc_kernel=enc_kernel,
                enc_dropout=enc_dropout, 
                window_size=window_size, 
                n_feats=n_feats, 
                dec_dim=dec_dim, 
                beta_min=beta_min, 
                beta_max=beta_max, 
                pe_scale=pe_scale
            ).to(device=device)
        else:
            print("Using GradTTS with Speaker Embedding and SALN model")
            generator = GradTTSWithSpeakerEmbeddingAndSALN(
                n_vocab=nsymbols,
                n_spks=2,
                spk_emb_dim=512,
                n_enc_channels=n_enc_channels,
                filter_channels=filter_channels,
                filter_channels_dp=filter_channels_dp,
                n_heads=n_heads,
                n_enc_layers=n_enc_layers,
                enc_kernel=enc_kernel,
                enc_dropout=enc_dropout, 
                window_size=window_size, 
                n_feats=n_feats, 
                dec_dim=dec_dim, 
                beta_min=beta_min, 
                beta_max=beta_max, 
                pe_scale=pe_scale
            ).to(device=device)
    generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc)["model_state_dict"])
    _ = generator.to(device=device).eval()
    print(f"Number of parameters: {generator.nparams}")
    
    print("Initializing HiFi-GAN...")
    if multi_speaker:
        HIFIGAN_CHECKPT = "./checkpts/generator_universal.pth.tar"
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"])
    _ = vocoder.to(device=device).eval()
    vocoder.remove_weight_norm()

    cmu = cmudict.CMUDict("./resources/cmu_dictionary")

    with torch.no_grad():
        for speaker in speaker_to_texts_and_filenames:
            print(f"Processing speaker: {speaker}")
            speaker_save_dir = os.path.join(save_dir, speaker)
            if not os.path.exists(speaker_save_dir):
                os.makedirs(speaker_save_dir, exist_ok=True)
            os.makedirs(speaker_save_dir, exist_ok=True)

            speaker_embedding_path = os.path.join(speaker_embedding_dir, dataset_name, f"{speaker}.npy")
            if not os.path.isfile(speaker_embedding_path):
                print(f"Speaker embedding for speaker {speaker} not found at {speaker_embedding_path}. Skipping...")
                continue
            speaker_embedding = torch.from_numpy(np.load(speaker_embedding_path)).to(device=device)

            texts_and_filenames = speaker_to_texts_and_filenames[speaker]
            for item in tqdm(texts_and_filenames, desc=f"Synthesizing for speaker {speaker}"):
                text = item["text"]
                file_name = item["file_name"]

                x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device=device)[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)

                y_enc, y_dec, attn = generator.forward(
                    x, 
                    x_lengths, 
                    n_timesteps=timesteps, 
                    temperature=1.5,
                    stoc=False, 
                    spk=speaker_embedding, 
                    length_scale=1.1
                )

                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                output_path = os.path.join(speaker_save_dir, file_name)
                write(output_path, params.sample_rate, audio)