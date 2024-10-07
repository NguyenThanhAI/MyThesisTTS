# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import os

import argparse

import time

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--audio_directory", type=str, default=params.audio_directory)
    parser.add_argument("--train_filelist_path", type=str, default=params.train_filelist_path)
    parser.add_argument("--valid_filelist_path", type=str, default=params.valid_filelist_path)
    parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--log_dir", type=str, default=params.log_dir)
    parser.add_argument("--n_epochs", type=int, default=params.n_epochs)
    parser.add_argument("--batch_size", type=int, default=params.batch_size)
    parser.add_argument("--out_size", type=int, default=params.out_size)
    parser.add_argument("--learning_rate", type=float, default=params.learning_rate)
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
    parser.add_argument("--n_fft", type=int, default=params.n_fft)
    parser.add_argument("--sample_rate", type=int, default=params.sample_rate)
    parser.add_argument("--hop_length", type=int, default=params.hop_length)
    parser.add_argument("--win_length", type=int, default=params.win_length)
    parser.add_argument("--f_min", type=int, default=params.f_min)
    parser.add_argument("--f_max", type=int, default=params.f_max)
    parser.add_argument("--dec_dim", type=int, default=params.dec_dim)
    parser.add_argument("--beta_min", type=float, default=params.beta_min)
    parser.add_argument("--beta_max", type=float, default=params.beta_max)
    parser.add_argument("--pe_scale", type=int, default=params.pe_scale)
    parser.add_argument("--save_every", type=int, default=params.save_every)
    parser.add_argument("--max_time_run", type=int, default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time.time()

    args = get_args()
    
    pretrained_checkpoint = args.pretrained_checkpoint
    audio_directory = args.audio_directory
    train_filelist_path = args.train_filelist_path
    valid_filelist_path = args.valid_filelist_path
    cmudict_path = args.cmudict_path
    add_blank = args.add_blank

    log_dir = args.log_dir
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    out_size = args.out_size
    learning_rate = args.learning_rate
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
    n_fft = args.n_fft
    sample_rate = args.sample_rate
    hop_length = args.hop_length
    win_length = args.win_length
    f_min = args.f_min
    f_max = args.f_max

    dec_dim = args.dec_dim
    beta_min = args.beta_min
    beta_max = args.beta_max
    pe_scale = args.pe_scale
    save_every = args.save_every
    max_time_run = args.max_time_run

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"Arguments: {args}")

    print("Initializing logger...")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    print("Initializing data loaders...")
    train_dataset = TextMelDataset(filelist_path=train_filelist_path, audio_directory=audio_directory,
                                   cmudict_path=cmudict_path, add_blank=add_blank,
                                   n_fft=n_fft, n_mels=n_feats, sample_rate=sample_rate, hop_length=hop_length,
                                   win_length=win_length, f_min=f_min, f_max=f_max)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelDataset(filelist_path=valid_filelist_path, audio_directory=audio_directory,
                                  cmudict_path=cmudict_path, add_blank=add_blank,
                                  n_fft=n_fft, n_mels=n_feats, sample_rate=sample_rate, hop_length=hop_length,
                                  win_length=win_length, f_min=f_min, f_max=f_max)
    
    print("Initializing model...")
    model = GradTTS(n_vocab=nsymbols, n_spks=1, spk_emb_dim=None, n_enc_channels=n_enc_channels, filter_channels=filter_channels, filter_channels_dp=filter_channels_dp, 
                    n_heads=n_heads, n_enc_layers=n_enc_layers, enc_kernel=enc_kernel, enc_dropout=enc_dropout, window_size=window_size, 
                    n_feats=n_feats, dec_dim=dec_dim, beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale).to(device=device)
    print("Number of encoder + duration predictor parameters: %.2fm" % (model.encoder.nparams/1e6))
    print("Number of decoder parameters: %.2fm" % (model.decoder.nparams/1e6))
    print("Total parameters: %.2fm" % (model.nparams/1e6))

    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if pretrained_checkpoint is not None:
        print("Load checkpoint")
        checkpoint = torch.load(pretrained_checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_done = checkpoint["epoch"]
        iteration = checkpoint["iteration"]
    else:
        epoch_done = 0
        iteration = 0

    print("Logging test batch...")
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item["y"]
        logger.add_image(f"image_{i}/ground_truth", plot_tensor(mel.squeeze()),
                         global_step=0, dataformats="HWC")
        save_plot(mel.squeeze(), f"{log_dir}/original_{i}.png")

    print("Start training...")
    # iteration = 0
    for epoch in range(epoch_done + 1, n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
                y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
                dur_loss, prior_loss, diff_loss = model.compute_loss(x=x, x_lengths=x_lengths,
                                                                     y=y, y_lengths=y_lengths,
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar("training/duration_loss", dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/prior_loss", prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/diffusion_loss", diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/encoder_grad_norm", enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar("training/decoder_grad_norm", dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f"Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}"
                    progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = "Epoch %d: duration loss = %.3f " % (epoch, np.mean(dur_losses))
        log_msg += "| prior loss = %.3f " % np.mean(prior_losses)
        log_msg += "| diffusion loss = %.3f\n" % np.mean(diff_losses)
        with open(f"{log_dir}/train.log", "a") as f:
            f.write(log_msg)

        time_run = time.time() - start_time
        
        if max_time_run is None:
            stop_now = False
        else:
            stop_now = True if time_run >= max_time_run else False

        if not stop_now:
            if epoch % save_every:
                continue

        model.eval()
        print("Synthesis...")
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item["x"].to(torch.long).unsqueeze(0).to(device=device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
                logger.add_image(f"image_{i}/generated_enc",
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats="HWC")
                logger.add_image(f"image_{i}/generated_dec",
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats="HWC")
                logger.add_image(f"image_{i}/alignment",
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats="HWC")
                save_plot(y_enc.squeeze().cpu(), 
                          f"{log_dir}/generated_enc_{i}.png")
                save_plot(y_dec.squeeze().cpu(), 
                          f"{log_dir}/generated_dec_{i}.png")
                save_plot(attn.squeeze().cpu(), 
                          f"{log_dir}/alignment_{i}.png")

        # ckpt = model.state_dict()
        ckpt = {"model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "iteration": iteration}
        print("Save check point at epoch {} and iteration {}".format(epoch, iteration))
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

        if stop_now:
            print("[INFO] Running out of time, stop training now")
            break
