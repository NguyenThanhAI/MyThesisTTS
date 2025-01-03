import os

import argparse

from copy import deepcopy

import time
import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import VarianceAdaptorGradTTS
from data_precomputed import PrecomputedTextMelDurPitchDataset, PrecomputedTextMelDurPitchBatchCollate
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
    # parser.add_argument("--audio_directory", type=str, default=params.audio_directory)
    # parser.add_argument("--train_filelist_path", type=str, default=params.train_filelist_path)
    # parser.add_argument("--valid_filelist_path", type=str, default=params.valid_filelist_path)
    # parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--log_dir", type=str, default="Saved_MyTTS/")
    parser.add_argument("--n_epochs", type=int, default=params.n_epochs)
    parser.add_argument("--batch_size", type=int, default=params.batch_size)
    parser.add_argument("--out_size", type=int, default=params.out_size)
    parser.add_argument("--learning_rate", type=float, default=params.learning_rate)
    parser.add_argument("--random_seed", type=int, default=params.seed)
    # parser.add_argument("nsymbols", type=int, default=len(symbols))
    parser.add_argument("--n_enc_channels", type=int, default=params.n_enc_channels)
    parser.add_argument("--filter_channels", type=int, default=params.filter_channels)
    # parser.add_argument("--filter_channels_dp", type=int, default=params.filter_channels_dp)
    parser.add_argument("--n_enc_layers", type=int, default=params.n_enc_layers)
    parser.add_argument("--enc_kernel", type=int, default=params.enc_kernel)
    parser.add_argument("--enc_dropout", type=float, default=params.enc_dropout)
    parser.add_argument("--n_heads", type=int, default=params.n_heads)
    parser.add_argument("--window_size", type=int, default=params.window_size)
    parser.add_argument("--n_feats", type=int, default=params.n_feats)
    # parser.add_argument("--n_fft", type=int, default=params.n_fft)
    # parser.add_argument("--sample_rate", type=int, default=params.sample_rate)
    # parser.add_argument("--hop_length", type=int, default=params.hop_length)
    # parser.add_argument("--win_length", type=int, default=params.win_length)
    # parser.add_argument("--f_min", type=int, default=params.f_min)
    # parser.add_argument("--f_max", type=int, default=params.f_max)
    parser.add_argument("--dec_dim", type=int, default=params.dec_dim)
    parser.add_argument("--beta_min", type=float, default=params.beta_min)
    parser.add_argument("--beta_max", type=float, default=params.beta_max)
    parser.add_argument("--pe_scale", type=int, default=params.pe_scale)
    parser.add_argument("--save_every", type=int, default=params.save_every)
    parser.add_argument("--max_time_run", type=int, default=None)

    parser.add_argument("--data_dir", type=str, default=params.data_dir)
    parser.add_argument("--dataset_name", type=str, default=params.dataset_name)
    
    parser.add_argument("--pitch_feature_level", type=str, default=params.pitch_feature_level)
    parser.add_argument("--pitch_quantization", type=str, default=params.pitch_quantization)
    parser.add_argument("--energy_feature_level", type=str, default=params.energy_feature_level)
    parser.add_argument("--energy_quantization", type=str, default=params.energy_quantization)
    parser.add_argument("--variance_dims", type=int, default=params.variance_dims)
    parser.add_argument("--stats_file_path", type=str, default=params.stats_file_path)
    parser.add_argument("--n_bins", type=int, default=params.n_bins)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time.time()

    args = get_args()
    
    pretrained_checkpoint = args.pretrained_checkpoint
    # audio_directory = args.audio_directory
    # train_filelist_path = args.train_filelist_path
    # valid_filelist_path = args.valid_filelist_path
    # cmudict_path = args.cmudict_path
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
    # filter_channels_dp = args.filter_channels_dp
    n_enc_layers = args.n_enc_layers
    enc_kernel = args.enc_kernel
    enc_dropout = args.enc_dropout
    n_heads = args.n_heads
    window_size = args.window_size

    n_feats = args.n_feats
    # n_fft = args.n_fft
    # sample_rate = args.sample_rate
    # hop_length = args.hop_length
    # win_length = args.win_length
    # f_min = args.f_min
    # f_max = args.f_max

    dec_dim = args.dec_dim
    beta_min = args.beta_min
    beta_max = args.beta_max
    pe_scale = args.pe_scale
    save_every = args.save_every
    max_time_run = args.max_time_run

    data_dir = args.data_dir
    dataset_name = args.dataset_name

    pitch_feature_level = args.pitch_feature_level
    pitch_quantization = args.pitch_quantization
    energy_feature_level = args.energy_feature_level
    energy_quantization = args.energy_quantization
    variance_dims = args.variance_dims
    stats_file_path = args.stats_file_path
    n_bins = args.n_bins

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"Arguments: {args}")

    print("Initializing logger...")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    print("Initializing data loaders...")

    train_dataset = PrecomputedTextMelDurPitchDataset(data_dir=data_dir, dataset_name=dataset_name, is_train=True)

    batch_collate = PrecomputedTextMelDurPitchBatchCollate()

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              collate_fn=batch_collate,
                              drop_last=True,
                              num_workers=4,
                              shuffle=True)
    
    test_dataset = PrecomputedTextMelDurPitchDataset(data_dir=data_dir, dataset_name=dataset_name, is_train=False)

    model = VarianceAdaptorGradTTS(n_vocab=nsymbols,
                                   n_enc_channels=n_enc_channels,
                                   filter_channels=filter_channels,
                                   n_heads=n_heads,
                                   n_enc_layers=n_enc_layers,
                                   enc_kernel=enc_kernel,
                                   enc_dropout=enc_dropout,
                                   window_size=window_size,
                                   n_feats=n_feats,
                                   dec_dim=dec_dim,
                                   beta_min=beta_min,
                                   beta_max=beta_max,
                                   pe_scale=pe_scale,
                                   pitch_feature_level=pitch_feature_level,
                                   pitch_quantization=pitch_quantization,
                                   energy_feature_level=energy_feature_level,
                                   energy_quantization=energy_quantization,
                                   variance_dims=variance_dims,
                                   stats_file_path=stats_file_path,
                                   n_bins=n_bins).to(device=device)
    
    print("Number of pre-encoder parameters: %.2fm" % (model.pre_encoder.nparams/1e6))
    print("Number of post-encoder parameters: %.2fm" % (model.post_encoder.nparams/1e6))
    print("Number of variance adaptor parameters: %.2fm" % (model.variance_adaptor.nparams/1e6))
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

    for epoch in range(epoch_done + 1, n_epochs + 1):
        model.train()

        dur_losses = []
        prior_losses = []
        diff_losses = []
        pitch_losses = []
        energy_losses = []

        with tqdm(train_loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                # model.zero_grad()
                optimizer.zero_grad()
                x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
                y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
                duration_target, pitch_target, energy_target = batch["duration"].to(device=device), batch["pitch"].to(device=device), batch["energy"].to(device=device)

                large_total_loss, mel_loss, pitch_loss, energy_loss, dur_loss, diff_loss = model.compute_loss(x=x,
                                                                                                              x_lengths=x_lengths,
                                                                                                              y=y,
                                                                                                              y_lengths=y_lengths,
                                                                                                              duration_target=duration_target,
                                                                                                              pitch_target=pitch_target,
                                                                                                              energy_target=energy_target,
                                                                                                              out_size=out_size)
                
                loss = sum([mel_loss, pitch_loss, energy_loss, dur_loss, diff_loss])

                # assert torch.abs(loss - large_total_loss).cpu().numpy() < 1e-5
                loss.backward()
                
                # print("======================================")
                # for name, param in model.named_parameters():
                #     print(name, param.grad)

                pre_enc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.pre_encoder.parameters(),
                                                                   max_norm=1)
                
                post_enc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.post_encoder.parameters(),
                                                                    max_norm=1)
                
                variance_adaptor_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.variance_adaptor.parameters(),
                                                                            max_norm=1)
                
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.decoder.parameters(),
                                                               max_norm=1)
                
                optimizer.step()

                logger.add_scalar("training/duration_loss", dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/mel_loss", mel_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/diffusion_loss", diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/pitch_loss", pitch_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/energy_loss", energy_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/pre_encoder_grad_norm", pre_enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar("training/post_encoder_grad_norm", post_enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar("training/variance_adaptor_grad_norm", variance_adaptor_grad_norm,
                                  global_step=iteration)
                logger.add_scalar("training/decoder_grad_norm", dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(mel_loss.item())
                pitch_losses.append(pitch_loss.item())
                energy_losses.append(energy_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 1 == 0:
                    msg = f"Epoch: {epoch}, iteration: {iteration} duration loss: {dur_loss.item():.3f}, mel loss: {mel_loss.item():.3f} pitch loss: {pitch_loss.item():.3f}, energy loss: {energy_loss.item():.3f} diff loss: {diff_loss.item():.3f}"
                    # print(msg)
                    progress_bar.set_description(msg)

                iteration += 1

                # if iteration >= 1:
                #     break

        log_msg = "Epoch %d, duration loss = %.3f" % (epoch, np.mean(dur_losses))
        log_msg += "| mel loss = %.3f" % np.mean(prior_losses)
        log_msg += "| pitch loss = %.3f" % np.mean(pitch_losses)
        log_msg += "| energy loss = %.3f" % np.mean(energy_losses)
        log_msg += "diffusion loss = %.3f" % np.mean(diff_losses)

        with open(os.path.join(log_dir, "train.log"), "a") as f:
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
        print("Synthesis")
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item["x"].to(torch.long).unsqueeze(0).to(device=device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
                y_enc, y_dec = model(x=x,
                                     x_lengths=x_lengths,
                                     n_timesteps=50)
                logger.add_image(f"image_{i}/generated_enc",
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats="HWC")
                logger.add_image(f"image_{i}/generated_dec",
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats="HWC")
                save_plot(y_enc.squeeze().cpu(), 
                          os.path.join(log_dir, f"generated_enc_{i}.png"))
                save_plot(y_dec.squeeze().cpu(), 
                          os.path.join(log_dir, f"generated_dec_{i}.png"))
                
        ckpt = {"model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "iteration": iteration}
        print("Save check point at epoch {} and iteration {}".format(epoch, iteration))
        torch.save(ckpt, f=os.path.join(log_dir, f"my_thesis_tts_{epoch}.pt"))

        if stop_now:
            print("[INFO] Running out of time, stop training now")
            break
