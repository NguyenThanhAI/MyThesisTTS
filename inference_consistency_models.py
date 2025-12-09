import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import ConsistencyModelWithSpeakerEmbeddingAdditive, ConsistencyModelWithSpeakerEmbeddingAndSALN
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default=r"D:\LJSpeech\text.txt", help="path to a file with texts to synthesize")
    parser.add_argument("-c", "--checkpoint", type=str, default=r"C:\Users\Thanh\Downloads\consistency_multi_speaker_LJSpeech_additive_steps_233793.pt", help="path to a checkpoint of Grad-TTS")
    parser.add_argument("-t", "--timesteps", type=int, required=False, default=5, help="number of timesteps of reverse diffusion")
    parser.add_argument("-s", "--speaker_style_wav", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Phoneme_Mel_Speaker_Embed\LJSpeech\speaker_embed\LJSpeech.npy", help="speaker id for multispeaker model")

    parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--random_seed", type=int, default=params.seed)
    parser.add_argument("--n_enc_channels", type=int, default=params.n_enc_channels)
    parser.add_argument("--filter_channels", type=int, default=params.filter_channels)
    parser.add_argument("--filter_channels_dp", type=int, default=params.filter_channels_dp)
    parser.add_argument("--n_enc_layers", type=int, default=params.n_enc_layers)
    parser.add_argument("--enc_kernel", type=int, default=params.enc_kernel)
    parser.add_argument("--enc_dropout", type=float, default=params.enc_dropout)
    parser.add_argument("--n_heads", type=int, default=params.n_heads)
    parser.add_argument("--window_size", type=int, default=params.window_size)
    parser.add_argument("--n_feats", type=int, default=params.n_feats)
    parser.add_argument("--dec_dim", type=int, default=64)

    parser.add_argument("--pe_scale", type=int, default=10) # Mặc địch là 10

    parser.add_argument("--use_additive", type=str2bool, default=True)

    parser.add_argument("--num_dec_blocks", type=int, default=20)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)

    parser.add_argument("--multi_speaker", type=str2bool, default=True)
    args = parser.parse_args()

    speaker_style_wav = args.speaker_style_wav

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
    pe_scale = args.pe_scale

    num_dec_blocks = args.num_dec_blocks
    sigma_max = args.sigma_max
    sigma_min = args.sigma_min
    rho = args.rho
    sigma_data = args.sigma_data

    use_additive = args.use_additive

    multi_speaker = args.multi_speaker
    
    style_vector = np.load(args.speaker_style_wav)
    style_vector = torch.from_numpy(style_vector).to(device=device)
    print("Initializing model...")

    if use_additive:

        print("Using Consistency Model with Speaker Embedding Additive model")
        generator = ConsistencyModelWithSpeakerEmbeddingAdditive(
            n_vocab=nsymbols,
            n_feats=n_feats,
            n_enc_channels=n_enc_channels,
            filter_channels=filter_channels,
            filter_channels_dp=filter_channels_dp,
            n_heads=n_heads,
            n_enc_layers=n_enc_layers,
            enc_kernel_size=enc_kernel,
            enc_dropout=enc_dropout,
            window_size=window_size,
            spk_emb_dim=512,
            dec_dim=dec_dim,
            num_dec_blocks=num_dec_blocks,
            pe_scale=pe_scale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            sigma_data=sigma_data,
            num_warmup_steps=1000,
            total_steps=1000000,
            start_ema_rate=0.9
        ).to(device)
    else:
        print("Using Consistency Model with Speaker Embedding and SALN model")
        generator = ConsistencyModelWithSpeakerEmbeddingAndSALN(
            n_vocab=nsymbols,
            n_feats=n_feats,
            n_enc_channels=n_enc_channels,
            filter_channels=filter_channels,
            filter_channels_dp=filter_channels_dp,
            n_heads=n_heads,
            n_enc_layers=n_enc_layers,
            enc_kernel_size=enc_kernel,
            enc_dropout=enc_dropout,
            window_size=window_size,
            spk_emb_dim=512,
            dec_dim=dec_dim,
            num_dec_blocks=num_dec_blocks,
            pe_scale=pe_scale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            sigma_data=sigma_data,
            num_warmup_steps=1000,
            total_steps=1000000,
            start_ema_rate=0.9
        ).to(device)

    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc)["model_state_dict"])
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
    
    with open(args.file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    # texts = ["This is my thesis", "I want to play a game"]
    cmu = cmudict.CMUDict("./resources/cmu_dictionary")
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f"Synthesizing {i} text...", end=" ")
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device=device)[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=style_vector, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f"Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}")

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f"./out/sample_{i}.wav", 22050, audio)

    print("Done. Check out `out` folder for samples.")
