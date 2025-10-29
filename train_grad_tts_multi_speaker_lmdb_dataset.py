import os

import argparse

import time
import math

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

import comet_ml
from comet_ml import Experiment, ExistingExperiment

import params
from model import GradTTSWithSpeakerEmbedding
from data import LMDBTextMelSpeakerEmbedPrecomputedDataset, LMDBTextMelSpeakerEmbedPrecomputedBatchCollate
from utils import plot_mel, plot_tensor, save_plot, plot_mel_comet, plot_attn_comet
from utils import TensorBoardLoggerExperimentLikeComet
from text.symbols import symbols

from typing import Union
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimal_num_workers_and_prefetch_factor(batch_size: int=32, max_workers: int=8):
    cpu_count = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    if cpu_count > 12:
        num_workers = min(cpu_count, 8)
    else:
        # If low RAM
        if ram_gb < 8:
            max_workers = min(max_workers, 2)
        elif ram_gb < 16:
            max_workers = min(max_workers, 4)
        
        # Assign num workers
        num_workers = min(cpu_count, batch_size, max_workers)

    if ram_gb <= 20:
        prefetch_factor = 4
    else:
        prefetch_factor = 8
    
    print(f"Optimal number of workers: {num_workers} (CPU cores: {cpu_count}, RAM: {ram_gb:.1f} GB, Batch size: {batch_size}, Max workers limit: {max_workers}), Prefetch factor: {prefetch_factor}")
    
    return num_workers, prefetch_factor

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_scheduler(
    optimizer,
    scheduler_type="cosine",
    num_training_steps=10000,
    num_warmup_steps=500,
    **kwargs
):
    """
    Return scheduler warmup.
    Support:
      - cosine
      - linear
      - cosine_restart
      - step
      - exponential (incremental)
      - exp_step (block steps)
    """
    gamma = kwargs.get("gamma", 0.95)
    step_ratio = kwargs.get("step_ratio", 0.3)
    decay_steps = kwargs.get("decay_steps", 1000)  # riêng cho exp_step
    cycles = kwargs.get("cycles", 1)

    def lr_lambda(current_step: int):
        # --- Phase 1: Warmup ---
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # --- Phase 2: Sau warmup ---
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(progress, 1.0)

        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        elif scheduler_type == "linear":
            return 1.0 - progress

        elif scheduler_type == "cosine_restart":
            return 0.5 * (1.0 + math.cos(math.pi * ((progress * cycles) % 1.0)))

        elif scheduler_type == "step":
            n_steps = int(progress / step_ratio)
            return gamma ** n_steps

        elif scheduler_type == "exponential":
            decay_steps_total = num_training_steps - num_warmup_steps
            return gamma ** (progress * decay_steps_total)

        elif scheduler_type == "exp_step":
            # Giảm theo block step cố định
            step_after_warmup = current_step - num_warmup_steps
            n_decays = step_after_warmup // decay_steps
            return gamma ** n_decays

        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)

def find_resume_checkpoint(resume_checkpoint_dir):
    if resume_checkpoint_dir is not None:
        print(f"looking for resume checkpoint in {resume_checkpoint_dir}")
        all_checkpoints = []
        for dirs, _, files in os.walk(resume_checkpoint_dir):
            for file in files:
                if file.endswith(".pt"):
                    all_checkpoints.append(os.path.join(dirs, file))
        # all_checkpoints = [x for x in all_checkpoints if "model" in x]
        if len(all_checkpoints) == 0:
            print("no checkpoints found")
            return None
        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        max_step_checkpoint = all_checkpoints[-1]
        print(f"found resume checkpoint {max_step_checkpoint}")
        return max_step_checkpoint
    return None

def save_model(model, optimizer, scheduler, epoch, iteration, batch_index):
    ckpt = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "batch_index": batch_index}
    print("Save check point at epoch {} and iteration {}".format(epoch, iteration))
    torch.save(ckpt, f=os.path.join(log_dir, f"grad_tts_multi_speaker_ljspeech_steps_{iteration}.pt"))
    

def evaluate_losses(model: GradTTSWithSpeakerEmbedding, val_loader: DataLoader, experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], step: int):
    print("Evaluate losses")
    model.eval()
    dur_loss_accumlative = 0
    prior_loss_accumulative = 0
    diffusion_loss_accumulative = 0
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)

            dur_loss, prior_loss, diff_loss = model.compute_loss(x=x, x_lengths=x_lengths,
                                                                 y=y, y_lengths=y_lengths,
                                                                 spk=spker_embed,
                                                                 out_size=out_size)
            size_of_this_batch = x.shape[0]
            num_samples += size_of_this_batch

            dur_loss_accumlative += dur_loss.item() * size_of_this_batch
            prior_loss_accumulative += prior_loss.item() * size_of_this_batch
            diffusion_loss_accumulative += diff_loss.item() * size_of_this_batch


        val_dur_loss = dur_loss_accumlative / num_samples
        val_prior_loss = prior_loss_accumulative / num_samples
        val_diff_loss = diffusion_loss_accumulative / num_samples

        print(f"Evaluate at step: {step}, val duration loss: {val_dur_loss}, val prior loss: {val_prior_loss}, val diff loss: {val_diff_loss}")
        experiment.log_metric("val/duration_loss", val_dur_loss,
                                  step=step)
        experiment.log_metric("val/prior_loss", val_prior_loss,
                                step=step)
        experiment.log_metric("val/diffusion_loss", val_diff_loss,
                                step=step)
    model.train()
        
def synthesize_melspectrogram(model: GradTTSWithSpeakerEmbedding, val_dataset, experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], step: int):
    print("Synthesis")
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(val_dataset):
            if np.random.rand() < 0.1:
                x = item["x"].to(torch.long).unsqueeze(0).to(device=device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
                y = item["y"]
                spker_embed = item["spker_embed"].to(device=device)
                y_enc, y_dec, attn = model(x, x_lengths, spk=spker_embed, n_timesteps=50)

                fig_mel_gt = plot_mel_comet(y)
                experiment.log_figure(
                    figure_name=f"val/image_{i}/groundtruth_melspectrogram",
                    figure=fig_mel_gt,
                    step=step
                )
                plt.close(fig_mel_gt)

                fig_enc = plot_mel_comet(y_enc.squeeze().cpu())
                experiment.log_figure(
                    figure_name=f"val/image_{i}/generated_enc",
                    figure=fig_enc,
                    step=step
                )
                plt.close(fig_enc)

                fig_dec = plot_mel_comet(y_dec.squeeze().cpu())
                experiment.log_figure(
                    figure_name=f"val/image_{i}/generated_dec",
                    figure=fig_dec,
                    step=step
                )
                plt.close(fig_dec)

                fig_attn = plot_attn_comet(attn.squeeze().cpu())
                experiment.log_figure(
                    figure_name=f"val/image_{i}/alignment",
                    figure=fig_attn,
                    step=step
                )
                plt.close(fig_attn)

    model.train()

            
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\LJSpeech")
    parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--log_dir", type=str, default=params.log_dir)
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_size", type=int, default=params.out_size)
    parser.add_argument("--learning_rate", type=float, default=params.learning_rate)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
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
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--max_time_run", type=int, default=None)
    parser.add_argument("--synthesize_every", type=int, default=1000)
    parser.add_argument("--logger_type", type=str, default="comet", choices=["comet", "tensorboard"])
    parser.add_argument("--comet_api_key", type=str, default=None)
    parser.add_argument("--comet_existing_experiment_id", type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    start_time = time.time()

    args = get_args()

    pretrained_dir = args.pretrained_dir
    dataset_dir = args.dataset_dir

    cmudict_path = args.cmudict_path
    add_blank = args.add_blank

    log_dir = args.log_dir
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    out_size = args.out_size
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    num_warmup_steps = args.num_warmup_steps
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
    synthesize_every = args.synthesize_every
    max_time_run = args.max_time_run

    logger_type = args.logger_type
    comet_api_key = args.comet_api_key
    comet_existing_experiment_id = args.comet_existing_experiment_id

    print(f"Arguments: {args}")

    if logger_type == "comet":
        os.environ["COMET_API_KEY"] = comet_api_key

        comet_ml.login()

        if comet_existing_experiment_id is not None:
            experiment = ExistingExperiment(
                project_name="grad-tts-multi-speaker",
                workspace="thanh-nguy-n",
                experiment_key=comet_existing_experiment_id
            )
        else:
            experiment = Experiment(
                project_name="grad-tts-multi-speaker",
                workspace="thanh-nguy-n"
            )

    print("Initializing data loaders...")
    num_workers, prefetch_factor = get_optimal_num_workers_and_prefetch_factor(batch_size=batch_size, max_workers=16)
    train_dataset = LMDBTextMelSpeakerEmbedPrecomputedDataset(
        filename="train.txt",
        dataset_dir=dataset_dir
    )
    batch_collate = LMDBTextMelSpeakerEmbedPrecomputedBatchCollate()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        shuffle=True
    )
    total_training_steps = n_epochs * len(train_loader)
    val_dataset = LMDBTextMelSpeakerEmbedPrecomputedDataset(
        filename="val.txt",
        dataset_dir=dataset_dir
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        shuffle=False
    )
    print("Initializing model...")
    model = GradTTSWithSpeakerEmbedding(
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
    print("Number of encoder + duration predictor parameters: %.2fm" % (model.encoder.nparams/1e6))
    print("Number of decoder parameters: %.2fm" % (model.decoder.nparams/1e6))
    print("Total parameters: %.2fm" % (model.nparams/1e6))

    print("Initializing optimizer...")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    scheduler = get_scheduler(optimizer=optimizer, 
                              scheduler_type=lr_scheduler, 
                              num_training_steps=total_training_steps, 
                              num_warmup_steps=num_warmup_steps,
                              gamma=0.999, 
                              decay_steps=len(train_loader))
    
    pretrained_checkpoint = find_resume_checkpoint(resume_checkpoint_dir=pretrained_dir)

    if pretrained_checkpoint is not None:
        print("Load checkpoint")
        checkpoint = torch.load(pretrained_checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch_start = checkpoint["epoch"]
        iteration_start = checkpoint["iteration"]
        start_batch_index = checkpoint["batch_index"]
    else:
        epoch_start = 1
        iteration_start = 0
        start_batch_index = 0
    
    if logger_type == "tensorboard":
        experiment = TensorBoardLoggerExperimentLikeComet(log_dir=log_dir, start_step=iteration_start)
    outer_bar = tqdm(total=total_training_steps, desc="Training", position=0)
    outer_bar.n = iteration_start
    epoch = epoch_start
    iteration = iteration_start
    print("Start training")
    if max_time_run is not None:
        if max_time_run > 0:
            print(f"Time limit set to {max_time_run} seconds.")
            check_time_limit = True
            import time
            start_time = time.time()
    else:
        check_time_limit = False
    while True:
        if iteration < iteration_start:
            iteration += 1
            continue
    
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1, leave=False)
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        for batch_idx, batch in enumerate(train_loader):
            if epoch == epoch_start and batch_idx < start_batch_index:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)
            dur_loss, prior_loss, diff_loss = model.compute_loss(x=x, x_lengths=x_lengths,
                                                                 y=y, y_lengths=y_lengths,
                                                                 spk=spker_embed,
                                                                 out_size=out_size)
            
            loss = sum([dur_loss, prior_loss, diff_loss])
            loss.backward()

            enc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.encoder.parameters(),
                                                            max_norm=1)
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.decoder.parameters(),
                                                            max_norm=1)
            
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]

            experiment.log_metric("training/duration_loss", dur_loss.item(),
                                  step=iteration)
            experiment.log_metric("training/prior_loss", prior_loss.item(),
                                  step=iteration)
            experiment.log_metric("training/diffusion_loss", diff_loss.item(),
                                  step=iteration)
            experiment.log_metric("training/encoder_grad_norm", enc_grad_norm,
                                  step=iteration)
            experiment.log_metric("training/decoder_grad_norm", dec_grad_norm,
                                  step=iteration)
            experiment.log_metric("learning_rate", current_lr,
                                  step=iteration)

            dur_losses.append(dur_loss.item())
            prior_losses.append(prior_loss.item())
            diff_losses.append(diff_loss.item())

            iteration += 1
            description = f"dur_loss: {dur_loss.item():.3f}, prior_loss: {prior_loss.item():.3f}, diff_loss: {diff_loss.item():.3f}"
            outer_bar.set_description(description)
            outer_bar.update(1)

            inner_bar.update(1)

            if iteration % save_every == 0:
                save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epoch=epoch,
                       iteration=iteration,
                       batch_index=batch_idx)
                
            if iteration % synthesize_every == 0:
                synthesize_melspectrogram(model=model,
                                          val_dataset=val_dataset,
                                          experiment=experiment,
                                          step=iteration)
        
        epoch += 1
        evaluate_losses(
            model=model,
            val_loader=val_loader,
            experiment=experiment,
            step=iteration
        )
        end_time = time.time()
        if check_time_limit and (end_time - start_time) > max_time_run:
            print(f"Time limit of {max_time_run} seconds reached. Stopping training.")
            save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epoch=epoch,
                       iteration=iteration,
                       batch_index=batch_idx)
            torch.cuda.empty_cache()
            quit()

        if iteration >= total_training_steps:
            print(f"Finish training at step {iteration} >= {total_training_steps}")
            save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epoch=epoch,
                       iteration=iteration,
                       batch_index=batch_idx)
            torch.cuda.empty_cache()
            quit()
