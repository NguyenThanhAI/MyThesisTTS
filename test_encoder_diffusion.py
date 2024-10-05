import warnings
warnings. filterwarnings("ignore")

import torch
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from data import TextMelDataset, TextMelBatchCollate
from torch.utils.data import DataLoader
import params

from text.symbols import symbols
from model.utils import sequence_mask, generate_path, fix_len_compatibility
from model import monotonic_align


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_symbols = len(symbols) + 1 if params.add_blank else len(symbols)

    encoder = TextEncoder(n_vocab=n_symbols, 
                          n_feats=params.n_feats,
                          n_channels=params.n_enc_channels,
                          filter_channels=params.filter_channels,
                          filter_channels_dp=params.filter_channels_dp,
                          n_heads=params.n_heads,
                          n_layers=params.n_enc_layers,
                          kernel_size=params.enc_kernel,
                          p_dropout=params.enc_dropout,
                          window_size=params.window_size)
    
    encoder.to(device)

    decoder = Diffusion(n_feats=params.n_feats,
                        dim=params.dec_dim,
                        beta_min=params.beta_min,
                        beta_max=params.beta_max,
                        pe_scale=params.pe_scale)
    
    decoder.to(device)
    
    dataset = TextMelDataset(filelist_path=params.train_filelist_path,
                             audio_directory=params.audio_directory,
                             cmudict_path=params.cmudict_path,
                             n_fft=params.n_fft,
                             n_mels=params.n_feats,
                             sample_rate=params.sample_rate,
                             hop_length=params.hop_length,
                             win_length=params.win_length,
                             f_min=params.f_min,
                             f_max=params.f_max)
    
    batch_collate = TextMelBatchCollate()

    loader = DataLoader(dataset=dataset, batch_size=params.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    

    for batch in loader:
        x = batch["x"].to(device)
        x_lengths = batch["x_lengths"].to(device)
        y = batch["y"].to(device)
        y_lengths = batch["y_lengths"].to(device)
        print(f"x: {x.shape}, x_lengths: {x_lengths.shape}, y: {y.shape}, y_lengths: {y_lengths.shape}")
        mu, logw, x_mask = encoder(x=x, x_lengths=x_lengths)
        w = torch.exp(input=logw) * x_mask
        w_ceil = torch.ceil(input=w)
        print(f"mu: {mu.shape}, logw: {logw.shape}, w: {torch.sum(torch.ceil(torch.exp(logw)), dim=-1)}, x_mask: {x_mask.shape}")
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(length=y_max_length)
        y_mask = sequence_mask(length=y_lengths, max_length=y_max_length_).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(duration=w_ceil.squeeze(1), mask=attn_mask.squeeze(1)).unsqueeze(1)
        print(f"y_mask: {y_mask.shape}, attn_mask: {attn_mask.shape}, w_ceil: {w_ceil.shape}, attn: {attn.shape}")
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu.transpose(1, 2)).transpose(1, 2)
        print(f"mu_y: {mu_y.shape}")
        encoder_outputs = mu_y[:, :, :y_max_length]
        temperature = 1.0
        z = mu_y + torch.randn_like(input=mu_y, device=mu_y.device) / temperature
        print(f"encoder_outputs: {encoder_outputs.shape}, z: {z.shape}")
        decoder_outputs = decoder(z=z, mask=y_mask, mu=mu_y, n_timesteps=10, stoc=True, spk=None)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        print(f"decoder_outputs: {decoder_outputs.shape}")
        # break