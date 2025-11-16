import os

import argparse
import json

from copy import deepcopy

import matplotlib.pyplot as plt

import time
import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import comet_ml
from comet_ml import Experiment, ExistingExperiment

import params
from model import VarianceAdaptorGradTTS, StyleVarianceAdaptorGradTTS
from data_precomputed import PrecomputedTextMelDurPitchDataset, PrecomputedTextMelDurPitchBatchCollate
from data import LMDBTextMelPitchEnergySpeakerEmbedPrecomputedDataset, LMDBTextMelPitchEnergySpeakerEmbedPrecomputedBatchCollate
from utils import plot_tensor_with_pitch_energy, save_plot_with_pitch_energy, expand, plot_mel_comet, plot_mel_with_pitch_energy_comet
from utils import TensorBoardLoggerExperimentLikeComet
from text.symbols import symbols

from typing import Union, Tuple

from train_grad_tts_multi_speaker_lmdb_dataset import get_optimal_num_workers_and_prefetch_factor, str2bool, get_scheduler, find_resume_checkpoint
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


def save_model(model, optimizer, scheduler, epoch, iteration, batch_index, dataset_name="LJSpeech"):
    ckpt = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "batch_index": batch_index}
    print("Save check point at epoch {} and iteration {}".format(epoch, iteration))

    torch.save(ckpt, f=os.path.join(log_dir, f"my_tts_multi_speaker_{dataset_name}_steps_{iteration}.pt"))


def evaluate_losses(model: StyleVarianceAdaptorGradTTS, val_loader: DataLoader, experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], step: int):
    print("Evaluate losses")
    model.eval()
    dur_loss_accumlative = 0
    mel_loss_accumulative = 0
    pitch_loss_accumulative = 0
    energy_loss_accumulative = 0
    diffusion_loss_accumulative = 0
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            duration_target, pitch_target, energy_target = batch["duration"].to(device=device), batch["pitch"].to(device=device), batch["energy"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)

            large_total_loss, mel_loss, pitch_loss, energy_loss, dur_loss, diff_loss = model.compute_loss(x=x,
                                                                                                          x_lengths=x_lengths,
                                                                                                          y=y,
                                                                                                          y_lengths=y_lengths,
                                                                                                          duration_target=duration_target,
                                                                                                          pitch_target=pitch_target,
                                                                                                          energy_target=energy_target,
                                                                                                          spk=spker_embed)
            size_of_this_batch = x.shape[0]
            num_samples += size_of_this_batch

            dur_loss_accumlative += dur_loss.item() * size_of_this_batch
            mel_loss_accumulative += mel_loss.item() * size_of_this_batch
            pitch_loss_accumulative += pitch_loss.item() * size_of_this_batch
            energy_loss_accumulative += energy_loss.item() * size_of_this_batch
            diffusion_loss_accumulative += diff_loss.item() * size_of_this_batch


        val_dur_loss = dur_loss_accumlative / num_samples
        val_mel_loss = mel_loss_accumulative / num_samples
        val_pitch_loss = pitch_loss_accumulative / num_samples
        val_energy_loss = energy_loss_accumulative / num_samples
        val_diff_loss = diffusion_loss_accumulative / num_samples

        print(f"Evaluate at step: {step}, val duration loss: {val_dur_loss:.3f}, val prior loss: {val_mel_loss:.3f}, val pitch loss: {val_pitch_loss:.3f}, val energy loss: {val_energy_loss:.3f}, val diff loss: {val_diff_loss:.3f}")
        experiment.log_metric("duration_loss/val", val_dur_loss,
                               step=step)
        experiment.log_metric("mel_loss/val", val_mel_loss,
                               step=step)
        experiment.log_metric("pitch_loss/val", val_pitch_loss,
                               step=step)
        experiment.log_metric("energy_loss/val", val_energy_loss,
                               step=step)
        experiment.log_metric("diffusion_loss/val", val_diff_loss,
                               step=step)
    model.train()

    
def synthesize_melspectrogram(model: StyleVarianceAdaptorGradTTS, val_dataset, experiment: Union[Experiment, ExistingExperiment, TensorBoardLoggerExperimentLikeComet], step: int,
                              pitch_feature_level: str, energy_feature_level: str, stats: Tuple[float, float, float, float]):
    print("Synthesis")
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(val_dataset):
            if np.random.rand() < 0.1:
                x = item["x"].to(torch.long).unsqueeze(0).to(device=device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device=device)
                y = item["y"]
                spker_embed = item["spker_embed"].to(device=device)
                y_enc, y_dec, pitch_prediction, energy_prediction, duration_prediction = model(
                    x=x,
                    x_lengths=x_lengths,
                    n_timesteps=50,
                    spk=spker_embed
                )
                if pitch_feature_level == "phoneme_level":
                    pitch_predict = expand(pitch_prediction.squeeze().cpu().numpy(), durations=duration_prediction.squeeze().cpu().numpy())
                elif pitch_feature_level == "frame_level":
                    pitch_predict = pitch_prediction.squeeze().cpu().numpy()

                if energy_feature_level == "phoneme_level":
                    energy_predict = expand(energy_prediction.squeeze().cpu().numpy(), durations=duration_prediction.squeeze().cpu().numpy())
                elif energy_feature_level == "frame_level":
                    energy_predict = energy_prediction.squeeze().cpu().numpy()
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

                fig_dec = plot_mel_with_pitch_energy_comet(tensor=y_dec.squeeze().cpu(),
                                                           pitch_predict=pitch_predict,
                                                           energy_predict=energy_predict,
                                                           stats=stats)
                experiment.log_figure(
                    figure_name=f"val/image_{i}/generated_dec",
                    figure=fig_dec,
                    step=step
                )
                plt.close(fig_dec)
    model.train()
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_dir", type=str, default=None)
    # parser.add_argument("--audio_directory", type=str, default=params.audio_directory)
    # parser.add_argument("--train_filelist_path", type=str, default=params.train_filelist_path)
    # parser.add_argument("--valid_filelist_path", type=str, default=params.valid_filelist_path)
    # parser.add_argument("--cmudict_path", type=str, default=params.cmudict_path)
    parser.add_argument("--dataset_dir", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Phoneme_Mel_Pitch_Energy_Speaker_Embed\LJSpeech")
    # parser.add_argument("--add_blank", type=str2bool, default=params.add_blank)
    parser.add_argument("--log_dir", type=str, default="Saved_MyTTS/")
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_size", type=int, default=params.out_size)
    parser.add_argument("--learning_rate", type=float, default=params.learning_rate)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
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

    parser.add_argument("--synthesize_every", type=int, default=1000)
    parser.add_argument("--logger_type", type=str, default="comet", choices=["comet", "tensorboard"])
    parser.add_argument("--comet_api_key", type=str, default=None)
    parser.add_argument("--comet_existing_experiment_id", type=str, default=None)
    # parser.add_argument("--data_dir", type=str, default=params.data_dir)
    # parser.add_argument("--dataset_name", type=str, default=params.dataset_name)
    
    parser.add_argument("--pitch_feature_level", type=str, default=params.pitch_feature_level)
    parser.add_argument("--pitch_quantization", type=str, default=params.pitch_quantization)
    parser.add_argument("--energy_feature_level", type=str, default=params.energy_feature_level)
    parser.add_argument("--energy_quantization", type=str, default=params.energy_quantization)
    parser.add_argument("--variance_dims", type=int, default=params.variance_dims)
    parser.add_argument("--stats_file_path", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Phoneme_Mel_Pitch_Energy_Speaker_Embed\LJSpeech\stats.json")
    parser.add_argument("--n_bins", type=int, default=params.n_bins)

    parser.add_argument("--log_to_file_every", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, default="LJSpeech")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time.time()

    args = get_args()
    
    pretrained_dir = args.pretrained_dir
    dataset_dir = args.dataset_dir
    # audio_directory = args.audio_directory
    # train_filelist_path = args.train_filelist_path
    # valid_filelist_path = args.valid_filelist_path
    # cmudict_path = args.cmudict_path
    # add_blank = args.add_blank

    log_dir = args.log_dir
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    out_size = args.out_size
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    num_warmup_steps = args.num_warmup_steps
    random_seed = args.random_seed

    # nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    nsymbols = len(symbols)
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
    synthesize_every = args.synthesize_every
    max_time_run = args.max_time_run

    logger_type = args.logger_type
    comet_api_key = args.comet_api_key
    comet_existing_experiment_id = args.comet_existing_experiment_id

    pitch_feature_level = args.pitch_feature_level
    pitch_quantization = args.pitch_quantization
    energy_feature_level = args.energy_feature_level
    energy_quantization = args.energy_quantization
    variance_dims = args.variance_dims
    stats_file_path = args.stats_file_path
    n_bins = args.n_bins

    log_to_file_every = args.log_to_file_every
    dataset_name = args.dataset_name

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"Arguments: {args}")

    print("Initializing logger...")

    if logger_type == "comet":
        os.environ["COMET_API_KEY"] = comet_api_key

        comet_ml.login()

        if comet_existing_experiment_id is not None:
            experiment = ExistingExperiment(
                project_name="my-tts",
                workspace="thanh-nguy-n",
                experiment_key=comet_existing_experiment_id
            )
        else:
            experiment = Experiment(
                project_name="my-tts",
                workspace="thanh-nguy-n"
            )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


    print("Initializing data loaders...")

    # train_dataset = PrecomputedTextMelDurPitchDataset(data_dir=data_dir, dataset_name=dataset_name, is_train=True)

    # batch_collate = PrecomputedTextMelDurPitchBatchCollate()
    train_dataset = LMDBTextMelPitchEnergySpeakerEmbedPrecomputedDataset("train.txt", dataset_dir=dataset_dir)
    batch_collate = LMDBTextMelPitchEnergySpeakerEmbedPrecomputedBatchCollate(pitch_feature=pitch_feature_level,
                                                                              energy_feature=energy_feature_level)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              collate_fn=batch_collate,
                              drop_last=True,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=4,
                              persistent_workers=True)
    
    total_training_steps = n_epochs * len(train_loader)
    
    val_dataset = LMDBTextMelPitchEnergySpeakerEmbedPrecomputedDataset("val.txt", dataset_dir=dataset_dir)

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

    # model = VarianceAdaptorGradTTS(n_vocab=nsymbols,
    #                                n_enc_channels=n_enc_channels,
    #                                filter_channels=filter_channels,
    #                                n_heads=n_heads,
    #                                n_enc_layers=n_enc_layers,
    #                                enc_kernel=enc_kernel,
    #                                enc_dropout=enc_dropout,
    #                                window_size=window_size,
    #                                n_feats=n_feats,
    #                                dec_dim=dec_dim,
    #                                beta_min=beta_min,
    #                                beta_max=beta_max,
    #                                pe_scale=pe_scale,
    #                                pitch_feature_level=pitch_feature_level,
    #                                pitch_quantization=pitch_quantization,
    #                                energy_feature_level=energy_feature_level,
    #                                energy_quantization=energy_quantization,
    #                                variance_dims=variance_dims,
    #                                stats_file_path=stats_file_path,
    #                                n_bins=n_bins).to(device=device)
    model = StyleVarianceAdaptorGradTTS(
        n_vocab=nsymbols,
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
        stats_file_path=stats_file_path,
        n_bins=n_bins,
        pitch_feature_level=pitch_feature_level,
        pitch_quantization=pitch_quantization,
        energy_feature_level=energy_feature_level,
        energy_quantization=energy_quantization,
        variance_dims=variance_dims,
        spk_emb_dim=512,
    ).to(device=device)
    
    with open(stats_file_path, "r") as f:
        stats = json.load(f)
        pitch_min = stats["pitch"][0]
        pitch_max = stats["pitch"][1]
        energy_min = stats["energy"][0]
        energy_max = stats["energy"][1]
    
    print("Number of pre-encoder parameters: %.2fm" % (model.pre_encoder.nparams/1e6))
    print("Number of post-encoder parameters: %.2fm" % (model.post_encoder.nparams/1e6))
    print("Number of variance adaptor parameters: %.2fm" % (model.variance_adaptor.nparams/1e6))
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

        dur_losses = []
        prior_losses = []
        diff_losses = []
        pitch_losses = []
        energy_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if epoch == epoch_start and batch_idx < start_batch_index:
                continue
            optimizer.zero_grad()
            x, x_lengths = batch["x"].to(device=device), batch["x_lengths"].to(device=device)
            y, y_lengths = batch["y"].to(device=device), batch["y_lengths"].to(device=device)
            duration_target, pitch_target, energy_target = batch["duration"].to(device=device), batch["pitch"].to(device=device), batch["energy"].to(device=device)
            spker_embed = batch["spker_embed"].to(device=device)

            large_total_loss, mel_loss, pitch_loss, energy_loss, dur_loss, diff_loss = model.compute_loss(x=x,
                                                                                                          x_lengths=x_lengths,
                                                                                                          y=y,
                                                                                                          y_lengths=y_lengths,
                                                                                                          duration_target=duration_target,
                                                                                                          pitch_target=pitch_target,
                                                                                                          energy_target=energy_target,
                                                                                                          spk=spker_embed)
            
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

            if iteration % log_to_file_every == 0:
                experiment.log_metric("duration_loss/training", dur_loss.item(),
                                    step=iteration)
                experiment.log_metric("mel_loss/training", mel_loss.item(),
                                    step=iteration)
                experiment.log_metric("diffusion_loss/training", diff_loss.item(),
                                    step=iteration)
                experiment.log_metric("pitch_loss/training", pitch_loss.item(),
                                    step=iteration)
                experiment.log_metric("energy_loss/training", energy_loss.item(),
                                    step=iteration)
                experiment.log_metric("pre_encoder_grad_norm/training", pre_enc_grad_norm,
                                    step=iteration)
                experiment.log_metric("post_encoder_grad_norm/training", post_enc_grad_norm,
                                    step=iteration)
                experiment.log_metric("variance_adaptor_grad_norm/training", variance_adaptor_grad_norm,
                                    step=iteration)
                experiment.log_metric("decoder_grad_norm/training", dec_grad_norm,
                                    step=iteration)
            
            dur_losses.append(dur_loss.item())
            prior_losses.append(mel_loss.item())
            pitch_losses.append(pitch_loss.item())
            energy_losses.append(energy_loss.item())
            diff_losses.append(diff_loss.item())

            iteration += 1
            description = f"dur_loss: {dur_loss.item():.3f}, mel_loss: {mel_loss.item():.3f}, diff_loss: {diff_loss.item():.3f}, pitch_loss: {pitch_loss.item():.3f}, energy_loss: {energy_loss.item():.3f}"
            outer_bar.set_description(description)
            outer_bar.update(1)

            if iteration % save_every == 0:
                save_model(model=model,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           epoch=epoch,
                           iteration=iteration,
                           batch_index=batch_idx,
                           dataset_name=dataset_name)

            if iteration % synthesize_every == 0:
                synthesize_melspectrogram(model=model,
                                          val_dataset=val_dataset,
                                          experiment=experiment,
                                          step=iteration,
                                          pitch_feature_level=pitch_feature_level,
                                          energy_feature_level=energy_feature_level,
                                          stats=(pitch_min, pitch_max, energy_min, energy_max))

            # if iteration >= 1:
            #     break

        log_msg = "Epoch %d, duration loss = %.3f" % (epoch, np.mean(dur_losses))
        log_msg += "| mel loss = %.3f" % np.mean(prior_losses)
        log_msg += "| pitch loss = %.3f" % np.mean(pitch_losses)
        log_msg += "| energy loss = %.3f" % np.mean(energy_losses)
        log_msg += "diffusion loss = %.3f" % np.mean(diff_losses)

        with open(os.path.join(log_dir, "train.log"), "a") as f:
            f.write(log_msg)

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
                       batch_index=batch_idx,
                       dataset_name=dataset_name)
            torch.cuda.empty_cache()
            quit()

        if iteration >= total_training_steps:
            print(f"Finish training at step {iteration} >= {total_training_steps}")
            save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epoch=epoch,
                       iteration=iteration,
                       batch_index=batch_idx,
                       dataset_name=dataset_name)
            torch.cuda.empty_cache()
            quit()
