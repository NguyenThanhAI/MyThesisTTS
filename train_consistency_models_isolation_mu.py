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
from model import ConsistencyModelWithSpeakerEmbeddingAdditiveAndIsolation, ConsistencyModelWithSpeakerEmbeddingAndSALNAndIsolation
from data import LMDBTextMelSpeakerEmbedPrecomputedDataset, LMDBTextMelSpeakerEmbedPrecomputedBatchCollate
from utils import plot_mel, plot_tensor, save_plot, plot_mel_comet, plot_attn_comet
from utils import TensorBoardLoggerExperimentLikeComet
from utils import get_optimal_num_workers_and_prefetch_factor, str2bool, get_scheduler, find_resume_checkpoint
from text.symbols import symbols

from typing import Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, optimizer, scheduler, epoch, iteration, batch_index, log_dir, use_additive, dataset_name="LJSpeech"):
    ckpt = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "batch_index": batch_index}
    print("Save check point at epoch {} and iteration {}".format(epoch, iteration))
    if use_additive:
        add = "additive"
    else:
        add = "use_saln"
    torch.save(ckpt, f=os.path.join(log_dir, f"consistency_multi_speaker_isolation_mu_{dataset_name}_{add}_steps_{iteration}.pt"))


def evaluate_losses(
        model: ConsistencyModelWithSpeakerEmbeddingAdditiveAndIsolation, 
        val_loader: DataLoader, 
        experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], 
        step: int,
        out_size: int,
    ):
    print("Evaluate losses")
    model.eval()
    dur_loss_accumlative = 0
    prior_loss_accumulative = 0
    consistency_loss_accumulative = 0
    recon_loss_accumulative = 0
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)

            dur_loss, prior_loss, consistency_loss, recon_loss = model.compute_loss(
                x=x, x_lengths=x_lengths,
                y=y, y_lengths=y_lengths,
                step=step,
                spk=spker_embed,
                out_size=out_size
            )
            size_of_this_batch = x.shape[0]
            num_samples += size_of_this_batch

            dur_loss_accumlative += dur_loss.item() * size_of_this_batch
            prior_loss_accumulative += prior_loss.item() * size_of_this_batch
            consistency_loss_accumulative += consistency_loss.item() * size_of_this_batch
            recon_loss_accumulative += recon_loss.item() * size_of_this_batch


        val_dur_loss = dur_loss_accumlative / num_samples
        val_prior_loss = prior_loss_accumulative / num_samples
        val_consistency_loss = consistency_loss_accumulative / num_samples
        val_recon_loss = recon_loss_accumulative / num_samples

        print(f"Evaluate at step: {step}, val duration loss: {val_dur_loss:.3f}, val prior loss: {val_prior_loss:.3f}, val consistency loss: {val_consistency_loss:.3f}, val recon loss: {val_recon_loss:.3f}")
        experiment.log_metric("duration_loss/val", val_dur_loss,
                               step=step)
        experiment.log_metric("prior_loss/val", val_prior_loss,
                               step=step)
        experiment.log_metric("consistency_loss/val", val_consistency_loss,
                               step=step)
        experiment.log_metric("recon_loss/val", val_recon_loss,
                               step=step)
    model.train()


def synthesize_melspectrogram(
        model: ConsistencyModelWithSpeakerEmbeddingAdditiveAndIsolation, 
        val_dataset, 
        experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], step: int):
    print("Synthesis")
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(val_dataset):
            if np.random.rand() < 0.1:
                x = item["x"].to(torch.long).unsqueeze(0).to(device=device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
                y = item["y"]
                spker_embed = item["spker_embed"].to(device=device)
                y_enc, y_dec, attn = model.forward(
                    x=x, 
                    x_lengths=x_lengths, 
                    spk=spker_embed, 
                    n_timesteps=5
                )

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
    parser.add_argument("--dataset_dir", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Phoneme_Mel_Speaker_Embed\LJSpeech")
    parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--log_dir", type=str, default=params.log_dir)
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
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
    # parser.add_argument("--n_fft", type=int, default=params.n_fft)
    # parser.add_argument("--sample_rate", type=int, default=params.sample_rate)
    # parser.add_argument("--hop_length", type=int, default=params.hop_length)
    # parser.add_argument("--win_length", type=int, default=params.win_length)
    # parser.add_argument("--f_min", type=int, default=params.f_min)
    # parser.add_argument("--f_max", type=int, default=params.f_max)
    parser.add_argument("--dec_dim", type=int, default=256)
    # parser.add_argument("--beta_min", type=float, default=params.beta_min)
    # parser.add_argument("--beta_max", type=float, default=params.beta_max)
    parser.add_argument("--pe_scale", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--max_time_run", type=int, default=None)
    parser.add_argument("--synthesize_every", type=int, default=5)
    parser.add_argument("--logger_type", type=str, default="tensorboard", choices=["comet", "tensorboard"])
    parser.add_argument("--comet_api_key", type=str, default=None)
    parser.add_argument("--comet_existing_experiment_id", type=str, default=None)

    # parser.add_argument("--use_saln", type=str2bool, default=False)
    parser.add_argument("--use_additive", type=str2bool, default=False)
    parser.add_argument("--log_to_file_every", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="LJSpeech")

    parser.add_argument("--num_dec_blocks", type=int, default=20)
    parser.add_argument("--start_ema_rate", type=float, default=0.90)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--start_scales", type=float, default=3)
    parser.add_argument("--end_scales", type=float, default=200)
    parser.add_argument("--weight_schedule", type=str, default="karras")
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

    dec_dim = args.dec_dim
    pe_scale = args.pe_scale
    save_every = args.save_every
    synthesize_every = args.synthesize_every
    max_time_run = args.max_time_run

    logger_type = args.logger_type
    comet_api_key = args.comet_api_key
    comet_existing_experiment_id = args.comet_existing_experiment_id

    use_additive = args.use_additive
    log_to_file_every = args.log_to_file_every
    dataset_name = args.dataset_name

    num_dec_blocks = args.num_dec_blocks
    start_ema_rate = args.start_ema_rate
    sigma_max = args.sigma_max
    sigma_min = args.sigma_min
    rho = args.rho
    sigma_data = args.sigma_data
    start_scales = args.start_scales
    end_scales = args.end_scales
    weight_schedule = args.weight_schedule

    print(f"Arguments: {args}")

    if logger_type == "comet":
        os.environ["COMET_API_KEY"] = comet_api_key

        comet_ml.login()

        if comet_existing_experiment_id is not None:
            experiment = ExistingExperiment(
                project_name="consistency-isolation-mu",
                workspace="thanh-nguy-n",
                experiment_key=comet_existing_experiment_id
            )
        else:
            experiment = Experiment(
                project_name="consistency-isolation-mu",
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
    if use_additive:
        print("Using Consistency Model Isolation Mu with Speaker Embedding Additive model")
        model = ConsistencyModelWithSpeakerEmbeddingAdditiveAndIsolation(
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
            num_warmup_steps=num_warmup_steps,
            total_steps=total_training_steps,
            num_dec_blocks=num_dec_blocks,
            pe_scale=pe_scale,
            start_ema_rate=start_ema_rate,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            sigma_data=sigma_data,
            start_scales=start_scales,
            end_scales=end_scales,
            weight_schedule=weight_schedule
        ).to(device=device)
    else:
        print("Using Consistency Model Isolation Mu with Speaker Embedding and SALN model")
        model = ConsistencyModelWithSpeakerEmbeddingAndSALNAndIsolation(
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
            num_warmup_steps=num_warmup_steps,
            total_steps=total_training_steps,
            num_dec_blocks=num_dec_blocks,
            pe_scale=pe_scale,
            start_ema_rate=start_ema_rate,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            sigma_data=sigma_data,
            start_scales=start_scales,
            end_scales=end_scales,
            weight_schedule=weight_schedule
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
    
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if epoch == epoch_start and batch_idx < start_batch_index:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)
            dur_loss, prior_loss, consistency_loss, recon_loss = model.compute_loss(
                x=x, x_lengths=x_lengths,
                y=y, y_lengths=y_lengths,
                step=iteration,
                spk=spker_embed,
                out_size=out_size
            )

            loss = sum([dur_loss, prior_loss, consistency_loss, recon_loss])
            loss.backward()

            enc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.encoder.parameters(),
                                                            max_norm=1)
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.decoder.parameters(),
                                                            max_norm=1)
            
            optimizer.step()
            scheduler.step()

            model.update_ema_target_params()

            current_lr = scheduler.get_last_lr()[0]
            if iteration % log_to_file_every == 0:
                experiment.log_metric(
                    "duration_loss/training", dur_loss.item(),
                    step=iteration
                )
                experiment.log_metric(
                    "prior_loss/training", prior_loss.item(),
                    step=iteration
                )
                experiment.log_metric(
                    "consistency_loss/training", consistency_loss.item(),
                    step=iteration
                )
                experiment.log_metric(
                    "recon_loss/training", recon_loss.item(),
                    step=iteration
                )
                experiment.log_metric(
                    "encoder_grad_norm/training", enc_grad_norm,
                    step=iteration
                )
                experiment.log_metric(
                    "decoder_grad_norm/training", dec_grad_norm,
                    step=iteration)
                experiment.log_metric(
                    "learning_rate", current_lr,
                    step=iteration)
            
            iteration += 1
            description = f"dur_loss: {dur_loss.item():.3f}, prior_loss: {prior_loss.item():.3f}, consistency loss: {consistency_loss.item():.3f}, recon loss: {recon_loss.item():.3f}"
            outer_bar.set_description(description)
            outer_bar.update(1)

            inner_bar.update(1)

            if iteration % save_every == 0:
                save_model(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    iteration=iteration,
                    batch_index=batch_idx,
                    log_dir=log_dir,
                    use_additive=use_additive,
                    dataset_name=dataset_name
                )

            if iteration % synthesize_every == 0:
                synthesize_melspectrogram(
                    model=model,
                    val_dataset=val_dataset,
                    experiment=experiment,
                    step=iteration
                )
        epoch += 1
        evaluate_losses(
            model=model,
            val_loader=val_loader,
            experiment=experiment,
            step=iteration,
            out_size=out_size
        )
        end_time = time.time()
        if check_time_limit and (end_time - start_time) > max_time_run:
            print(f"Time limit of {max_time_run} seconds reached. Stopping training.")
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                iteration=iteration,
                batch_index=batch_idx,
                log_dir=log_dir,
                use_additive=use_additive,
                dataset_name=dataset_name
            )
            torch.cuda.empty_cache()
            quit()

        if iteration >= total_training_steps:
            print(f"Finish training at step {iteration} >= {total_training_steps}")
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                iteration=iteration,
                batch_index=batch_idx,
                log_dir=log_dir,
                use_additive=use_additive,
                dataset_name=dataset_name
            )
            torch.cuda.empty_cache()
            quit()