import os
from typing import List, Dict, Tuple

from tqdm import tqdm

import re
import time

import joblib
import librosa
import re
import numpy as np
import pandas as pd
import torch
import pyworld
import pysptk
import math
from fastdtw import fastdtw
from functools import partial
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
import sys
from speaker_embedder import PreDefinedEmbedder
from metrics.f0_frame_error import F0FrameError
from metrics.mos import MOSCal
from metrics.fid import CalFidSeries, CalRecall, CalPrecision, CalFIDAlign
import os.path as osp
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import librosa
from scipy.stats import entropy

import whisper
import jiwer
from pymcd.mcd import Calculate_MCD
from pydub import AudioSegment

import torchaudio as ta
from meldataset import mel_spectrogram


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


class MetricCalculator:
    def __init__(
            self,
            synthetized_data_dir: str,
            model_name: str,
            dataset_name: str,
            reference_data_dir: str,
            sampling_rate: int=22050,
            frame_period: float=5.,
        ):

        super(MetricCalculator, self).__init__()
        self.synthetized_data_dir = synthetized_data_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.reference_data_dir = reference_data_dir

        self.sampling_rate = sampling_rate
        self.frame_period = frame_period

        self.speakers_to_synth_wavs_and_reference = self.enumerate_synthesized_wavs_files_and_reference()

        self.mos_tool: MOSCal = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def enumerate_synthesized_wavs_files_and_reference(self) -> Dict[str, List[Dict[str, str]]]:
        syn_data_dir = os.path.join(self.synthetized_data_dir, self.model_name, self.dataset_name)
        speakers_to_synth_wavs_and_reference: Dict[str, List[Dict[str, str]]] = dict()
        for dirs, _, files in os.walk(syn_data_dir):
            for file in tqdm(files):
                if file.endswith(".wav"):
                    file_path = os.path.join(dirs, file)
                    speaker_id = os.path.dirname(file_path).split(os.sep)[-1]

                    if speaker_id not in speakers_to_synth_wavs_and_reference:
                        speakers_to_synth_wavs_and_reference[speaker_id] = []

                    possible_reference_file_path = os.path.join(self.reference_data_dir, self.dataset_name, speaker_id, file)
                    if os.path.exists(possible_reference_file_path):
                        speakers_to_synth_wavs_and_reference[speaker_id].append(
                            {
                                "synthesized_wav": file_path,
                                "reference_wav": possible_reference_file_path
                            }
                        )

        return speakers_to_synth_wavs_and_reference
    
    def _read_wav(self, file_path):
        return librosa.load(file_path, sr=self.sampling_rate, mono=True)[0]
    
    def _get_mgc(self, file_path):
        alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
        fft_size = 512
        mcep_size = 24
        wav = self._read_wav(file_path)
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=self.sampling_rate,
                                     frame_period=self.frame_period, fft_size=fft_size)
        # Extract MCEP features
        return pysptk.sptk.mcep(
            sp, order=mcep_size, alpha=alpha, maxiter=0,
            etype=1, eps=1.0E-8, min_det=0.0, itype=3
        )
    
    def _get_f0(self, wav_filepath):
        wav = self._read_wav(wav_filepath)
        wav = wav.astype(np.float64)
        f0, _ = pyworld.harvest(wav, self.sampling_rate, frame_period=self.frame_period, f0_floor=71.0, f0_ceil=800.0)
        return f0
    
    def _get_align_f0(self, synth_wav_path: str, ref_wav_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Get f0
        synth_f0 = self._get_f0(synth_wav_path)
        ref_f0 = self._get_f0(ref_wav_path)

        # Only select the voiced parts
        synth_f0 = synth_f0[synth_f0 > 0].reshape(1, -1)
        ref_f0 = ref_f0[ref_f0 > 0].reshape(1, -1)

        # Perform DTW alignment
        _, path = fastdtw(synth_f0.T, ref_f0.T)
        aligned_synth_f0 = synth_f0[:, [p[0] for p in path]].T.reshape(-1)
        aligned_ref_f0 = ref_f0[:, [p[1] for p in path]].T.reshape(-1)
        return aligned_synth_f0, aligned_ref_f0
    
    def _get_mfcc(self, filepath):
        mfcc = librosa.feature.mfcc(y=librosa.load(filepath)[0], sr=self.sampling_rate).T  # (seq_len,20)

        # Normalize the aligned MFCC features
        return mfcc / np.linalg.norm(mfcc, axis=0)  # (seq_len,20)
    

    @staticmethod
    def _get_gmm_kl(synth_wav_path, ref_wav_path, get_feature_fun):
        """
        This must have the label first and the prediction second.
        Reference link: http://t.csdnimg.cn/QZqQu
        :param get_feature_fun: 
        :param wav_filepath_pair:
        :return:
        """
        feature_target = get_feature_fun(synth_wav_path)  # (seq_len,20)
        gmm_target = GaussianMixture(n_components=30, covariance_type="full")
        gmm_target.fit(feature_target)

        feature_pre = get_feature_fun(ref_wav_path)  # (seq_len,20)
        gmm_pre = GaussianMixture(n_components=30, covariance_type="full")
        gmm_pre.fit(feature_pre)
        kl_ = entropy(gmm_target.score_samples(feature_target), gmm_pre.score_samples(feature_target))
        return 0 if kl_ == np.inf else kl_

    def compute_mfcc_gmm_kl(self):
        deal_pair_fun = partial(self._get_gmm_kl, get_feature_fun=self._get_mfcc)
        
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            kl_list = []
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing MFCC GMM KL for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                kl_value = deal_pair_fun(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                kl_list.append(kl_value)
        avg_kl = np.mean(np.array(kl_list)) if len(kl_list) > 0 else float("inf")
        return avg_kl
    
    def _get_align_fid_tool(self):
        fid_cal_tool = CalFIDAlign(
            speakers_to_synth_wavs_and_reference=self.speakers_to_synth_wavs_and_reference,
            sample_rate=self.sampling_rate,
        )
        return fid_cal_tool
    
    def compute_fid_align_mfcc(self):
        fid_cal_tool = self._get_align_fid_tool()
        return fid_cal_tool(feature_type="mfcc_un_norm", norm=True)
    
    def compute_fid_align_mfcc_un_norm(self):
        fid_cal_tool = self._get_align_fid_tool()
        return fid_cal_tool(feature_type="mfcc_un_norm", norm=False)
    
    def compute_fid_align_mel(self):
        fid_cal_tool = self._get_align_fid_tool()
        return fid_cal_tool(feature_type="mel", norm=False)
    
    def compute_wer_un_comma(self):
        model = whisper.load_model("medium", device=self.device)

        groundtruth_texts = []
        hypothesis_texts = []

        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing WER for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]

                # Transcribe synthesized audio
                try:
                    synth_result = model.transcribe(synth_wav_path)["text"][1:]
                except Exception as e:
                    synth_wav = self._read_wav(file_path=synth_wav_path)
                    synth_result = model.transcribe(synth_wav)["text"][1:]
                syn_text = synth_result.lower()
                syn_text = syn_text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")

                ref_text_path = ref_wav_path.replace(".wav", ".lab")
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()

                ref_text = ref_text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")

                groundtruth_texts.append(ref_text)
                hypothesis_texts.append(syn_text)

        wer = jiwer.wer(groundtruth_texts, hypothesis_texts)
        del model
        return wer
    
    def compute_wer(self):
        model = whisper.load_model("medium", device=self.device)

        groundtruth_texts = []
        hypothesis_texts = []

        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing WER for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                # synth_wav_path = Path(synth_wav_path).resolve().as_posix()
                # if os.path.exists(synth_wav_path) and os.path.isfile(synth_wav_path):
                #     print(True)
                # Transcribe synthesized audio
                try:
                    synth_result = model.transcribe(synth_wav_path)["text"][1:]
                except Exception as e:
                    synth_wav = self._read_wav(file_path=synth_wav_path)
                    synth_result = model.transcribe(synth_wav)["text"][1:]
                syn_text = synth_result.lower()

                ref_text_path = ref_wav_path.replace(".wav", ".lab")
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()


                groundtruth_texts.append(ref_text)
                hypothesis_texts.append(syn_text)

        wer = jiwer.wer(groundtruth_texts, hypothesis_texts)
        del model
        return wer
    
    def compute_si_sdr(self):
        def cal_pair(synth_wav_path, ref_wav_path):
            """Scale-Invariant Signal to Distortion Ratio (SI-SDR)
            :param filepath_pair: Must be in this order, the order matters here
            """
            y1 = librosa.load(synth_wav_path)[0].reshape(-1, 1)
            y2 = librosa.load(ref_wav_path)[0].reshape(-1, 1)
            f1 = y1.T  # (feature_dim,seq_len)
            f2 = y2.T  # (feature_dim,seq_len)
            # Use fastdtw to align the two MFCC feature matrices
            _, path = fastdtw(f1.T, f2.T)
            # Aligned feature matrices
            aligned_syn = f1[:, [p[0] for p in path]].T
            aligned_ref = f2[:, [p[1] for p in path]].T
            eps = np.finfo(float).eps  # A very small value to ensure it's not zero
            alpha = np.dot(aligned_syn.T, aligned_ref) / (np.dot(aligned_syn.T, aligned_syn) + eps)

            numerator = ((alpha * aligned_ref) ** 2).sum()  # numerator
            denominator = ((alpha * aligned_ref - aligned_syn) ** 2).sum()  # denominator

            return 10 * np.log10(numerator / (denominator + eps))

        si_sdr_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing SI-SDR for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                si_sdr_value = cal_pair(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                si_sdr_list.append(si_sdr_value)
        avg_si_sdr = np.mean(np.array(si_sdr_list)) if len(si_sdr_list) > 0 else float("-inf")
        return avg_si_sdr
    
    def compute_f0_corr(self):
        def cal_pair(synth_wav_path, ref_wav_path):
            aligned_synth_f0, aligned_ref_f0 = self._get_align_f0(synth_wav_path, ref_wav_path)
            f0corr = np.corrcoef(aligned_synth_f0, aligned_ref_f0)[0, 1]
            return f0corr
        
        f0corr_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing F0 CORR for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                f0corr_value = cal_pair(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                f0corr_list.append(f0corr_value)
        avg_f0corr = np.mean(np.array(f0corr_list)) if len(f0corr_list) > 0 else float("-inf")
        return avg_f0corr
        
    def compute_f0_rmse(self):
        def cal_pair(synth_wav_path, ref_wav_path):
            aligned_f0_1, aligned_f0_2 = self._get_align_f0(synth_wav_path, ref_wav_path)

            # only calculate f0 error for voiced frame
            y = 1200 * np.abs(np.log2(aligned_f0_1) - np.log2(aligned_f0_2))
            # print(y.sum(), tp_mask.sum())
            f0_rmse_mean = np.mean(y)
            # print(min_cost.shape)
            return f0_rmse_mean
        
        f0rmse_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing F0 RMSE for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                f0rmse_value = cal_pair(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                f0rmse_list.append(f0rmse_value)

        avg_f0rmse = np.mean(np.array(f0rmse_list)) if len(f0rmse_list) > 0 else float("inf")
        return avg_f0rmse
    
    def compute_log_f0(self):
        def cal_pair(synth_wav_path, ref_wav_path):
            f0_synth = self._get_mgc(file_path=synth_wav_path)
            f0_ref = self._get_mgc(file_path=ref_wav_path)

            # print(min(len(f0_synth), len(f0_ref)))
            def logf0_rmse(x, y):
                log_spec_dB_const = 1 / min(len(f0_synth), len(f0_ref))
                diff = x - y
                return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

            min_cost, _ = librosa.sequence.dtw(f0_synth[:, 1:].T, f0_ref[:, 1:].T, metric=logf0_rmse)
            # print(min_cost.shape)
            return np.mean(min_cost)
        
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            logf0rmse_list = []
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing Log F0 RMSE for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                logf0rmse_value = cal_pair(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                logf0rmse_list.append(logf0rmse_value)
        
        avg_logf0rmse = np.mean(np.array(logf0rmse_list)) if len(logf0rmse_list) > 0 else float("inf")
        return avg_logf0rmse
    

    def compute_ssim(self):
        sampling_rate = self.sampling_rate
        ssim_pair_cache = list()
        max_list = list()
        min_list = list()

        def find_ssim_max(audio_file1, audio_file2):
            # Extract MFCC features
            mfcc1 = librosa.feature.mfcc(y=librosa.load(audio_file1)[0], sr=sampling_rate)
            mfcc2 = librosa.feature.mfcc(y=librosa.load(audio_file2)[0], sr=sampling_rate)
            # Use fastdtw to align the two MFCC feature matrices
            _, path = fastdtw(mfcc1.T, mfcc2.T)
            # Aligned feature matrices
            aligned_mfcc1 = mfcc1[:, [p[0] for p in path]].T
            aligned_mfcc2 = mfcc2[:, [p[1] for p in path]].T
            # Normalize the aligned MFCC features
            aligned_mfcc1 = aligned_mfcc1 / np.linalg.norm(aligned_mfcc1, axis=0)
            aligned_mfcc2 = aligned_mfcc2 / np.linalg.norm(aligned_mfcc2, axis=0)
            max_ = max(np.max(aligned_mfcc1), np.max(aligned_mfcc2))
            min_ = min(np.min(aligned_mfcc1), np.min(aligned_mfcc2))
            max_list.append(max_)
            min_list.append(min_)
            ssim_pair_cache.append((aligned_mfcc1, aligned_mfcc2))

        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing SSIM for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                find_ssim_max(synth_wav_path, ref_wav_path)

        ssim = StructuralSimilarityIndexMeasure(data_range=max(max_list) - min(min_list))

        def cal_ssim(pair):
            return ssim(
                torch.unsqueeze(torch.unsqueeze(torch.from_numpy(pair[0]), 0), 0),
                torch.unsqueeze(torch.unsqueeze(torch.from_numpy(pair[1]), 0), 0)
            )

        return np.mean(np.array(list(map(cal_ssim, ssim_pair_cache))))
    

    def compute_mcd24(self):
        # SAMPLING_RATE = 22050
        # FRAME_PERIOD = 5.0
        alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
        fft_size = 512
        mcep_size = 24

        def log_spec_dB_dist(x, y):
            log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
            diff = x - y
            return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

        def wav2mcep_numpy(wavfile, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size, type=None):
            wav, _ = librosa.load(wavfile, sr=self.sampling_rate, mono=True)
            # Use WORLD vocoder to spectral envelope
            _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=self.sampling_rate,
                                         frame_period=self.frame_period, fft_size=fft_size)
            # Extract MCEP features
            mcep = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                                   etype=1, eps=1.0E-8, min_det=0.0, itype=3)
            # os.makedirs(os.path.join(self.mel_npy_path, "raw"), exist_ok=True)
            # os.makedirs(os.path.join(self.mel_npy_path, "syn"), exist_ok=True)
            # if "raw" in str(wavfile):
            #     mcep_name = os.path.join(self.mel_npy_path, "raw", str(wavfile).split("/")[-1].replace(".wav", ""))
            # else:
            #     mcep_name = os.path.join(self.mel_npy_path, "syn", str(wavfile).split("/")[-1].replace(".wav", ""))
            # np.save(mcep_name + ".npy", mgc, allow_pickle=False)
            # mcep_name = mcep_name + ".npy"
            # return mcep_name
            return mcep
        
        def average_mcd(synth_vec, ref_vec, cost_function):
            # min_cost_tot = 0.0
            # frames_tot = 0
            # synth_vec = np.load(syn_mcep_file)
            # ref_vec = np.load(raw_mcep_file)  # load MCEP vectors
            ref_frame_no = len(ref_vec)
            # dynamic time warping using librosa
            min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T,
                                               metric=cost_function)
            # min_cost_tot += np.mean(min_cost)
            # frames_tot += ref_frame_no
            # mean_mcd = min_cost_tot / frames_tot
            return min_cost, ref_frame_no
        
        mcd_mean = 0.0
        frames_used_toal = 0
        cost_function = log_spec_dB_dist
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing MCD24 for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                synth_mcep = wav2mcep_numpy(synth_wav_path)
                ref_mcep = wav2mcep_numpy(ref_wav_path)
                min_cost, ref_frame_no = average_mcd(synth_vec=synth_mcep, ref_vec=ref_mcep, cost_function=cost_function)
                mcd_mean += np.sum(min_cost)  # sum of all frame costs
                frames_used_toal += ref_frame_no

        mcd_result = mcd_mean / frames_used_toal if frames_used_toal > 0 else float("inf")
        return mcd_result
    
    def compute_mcd(self):
        
        def cal_pair(synth_wav_path, ref_wav_path):
            # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
            mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
            return mcd_toolbox.calculate_mcd(synthesized_audio=synth_wav_path, reference_audio=ref_wav_path)

        mcd_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing MCD for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                mcd_value = cal_pair(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                mcd_list.append(mcd_value)

        avg_mcd = np.mean(np.array(mcd_list)) if len(mcd_list) > 0 else float("inf")
        return avg_mcd
    
    def _compute_precision(self, feature_type: str):
        precision_cal_tool = CalPrecision(
            speakers_to_synth_wavs_and_reference=self.speakers_to_synth_wavs_and_reference,
            sample_rate=self.sampling_rate,
        )
        # if self.clear_cache:
        #     precision_cal_tool.clear_cache()
        return precision_cal_tool(feature_type=feature_type)
    
    def compute_precision_mel(self):
        return self._compute_precision("mel")

    def compute_precision_mfcc(self):
        return self._compute_precision("mfcc")
    
    def _compute_recall(self, feature_type):
        recall_cal_tool = CalRecall(
            speakers_to_synth_wavs_and_reference=self.speakers_to_synth_wavs_and_reference,
            sample_rate=self.sampling_rate,
        )
        # if self.clear_cache:
        #     recall_cal_tool.clear_cache()
        return recall_cal_tool(feature_type)
    
    def compute_recall_mfcc(self):
        return self._compute_recall("mfcc")

    def compute_recall_mel(self):
        return self._compute_recall("mel")
    
    def _compute_fid(self, feature_type):
        fid_cal_tool = CalFidSeries(
            speakers_to_synth_wavs_and_reference=self.speakers_to_synth_wavs_and_reference,
            sample_rate=self.sampling_rate,
        )
        # if self.clear_cache:
        #     fid_cal_tool.clear_cache()
        return fid_cal_tool(feature_type)
    
    def compute_fid_mfcc(self):
        return self._compute_fid("mfcc")

    def compute_fid_mfcc_un_norm(self):
        return self._compute_fid("mfcc_un_norm")

    def compute_fid_mel(self):
        return self._compute_fid("mel")
    
    def _mos_init(self):
        """
        Create the MOS tool only when used, and create it only once.
        """
        if self.mos_tool is None:
            self.mos_tool = MOSCal(sample_rate=self.sampling_rate)
    
    def _get_file_list_mean_mos(self, filename_list, mos_type="mb"):
        self._mos_init()
        if mos_type == "mb":
            return np.mean(np.array(list(map(self.mos_tool.get_mb_mos, tqdm(filename_list, desc=f"MB MOS")))))
        elif mos_type == "ld":
            return np.mean(np.array(list(map(self.mos_tool.get_ld_mos, tqdm(filename_list, desc=f"LD MOS")))))
        elif mos_type == "both":
            mb_mos = np.mean(np.array(list(map(self.mos_tool.get_mb_mos, tqdm(filename_list, desc=f"MB MOS")))))
            ld_mos = np.mean(np.array(list(map(self.mos_tool.get_ld_mos, tqdm(filename_list, desc=f"LD MOS")))))
            return (mb_mos + ld_mos) / 2.0
        else:
            raise NotImplementedError
        
    def compute_mb_mos(self):
        synth_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in wav_file_pairs:
                synth_wav_path = wav_file_pair["synthesized_wav"]
                synth_list.append(synth_wav_path)
        return self._get_file_list_mean_mos(filename_list=synth_list, mos_type="mb")

    def compute_ld_mos(self):
        synth_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in wav_file_pairs:
                synth_wav_path = wav_file_pair["synthesized_wav"]
                synth_list.append(synth_wav_path)
        return self._get_file_list_mean_mos(filename_list=synth_list, mos_type="ld")
    
    def compute_mb_ld_mos(self):
        synth_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in wav_file_pairs:
                synth_wav_path = wav_file_pair["synthesized_wav"]
                synth_list.append(synth_wav_path)
        return self._get_file_list_mean_mos(filename_list=synth_list, mos_type="both")
    
    def get_target_mos(self, mos_type):
        target_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in wav_file_pairs:
                ref_wav_path = wav_file_pair["reference_wav"]
                target_list.append(ref_wav_path)
        return self._get_file_list_mean_mos(filename_list=target_list, mos_type=mos_type)
    
    def compute_target_mb_mos(self):
        return self.get_target_mos(mos_type="mb")
    
    def compute_target_ld_mos(self):
        return self.get_target_mos(mos_type="ld")
    
    def compute_target_mb_ld_mos(self):
        return self.get_target_mos(mos_type="both")
    
    def compute_ffe(self):
        # SAMPLING_RATE = 22050
        trim_top_db = 23
        filter_length = 1024
        hop_length = 256

        def load_audio(wav_path):
            wav_raw, _ = librosa.load(wav_path, sr=self.sampling_rate)
            _, index = librosa.effects.trim(wav_raw, top_db=trim_top_db, frame_length=filter_length,
                                            hop_length=hop_length)
            wav = wav_raw[index[0]:index[1]]
            duration = (index[1] - index[0]) / hop_length
            return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

        ffe = F0FrameError(sr=self.sampling_rate)
        frame_error_rate = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing FFE for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                synth_wav_raw, synth_wav, synth_duration = load_audio(synth_wav_path)
                ref_wav_raw, ref_wav, ref_duration = load_audio(ref_wav_path)
                data = [ref_wav, synth_wav]
                data = pad_1D(data)
                ref_wav, synth_wav = data
                score = ffe.calculate_ffe(torch.tensor(ref_wav), torch.tensor(synth_wav))
                # print(score)
                frame_error_rate.append(score)
        return np.mean(frame_error_rate)  # , np.var(frame_error_rate)
    
    def compute_speaker_cos(self):
        def get_speaker_cos(synth_wav_path, ref_wav_path):
            def get_speaker_ebd(wav_filepath):
                def wav_to_16000(wav_filepath):
                    # if osp.exists(wav_filepath + "_16000"):
                    #     return wav_filepath + "_16000"
                    # else:
                        
                    audio = AudioSegment.from_wav(wav_filepath)
                    audio = audio.set_frame_rate(16000)
                    # audio.export(wav_filepath + "_16000")
                    # return wav_filepath + "_16000"
                    return audio

                # fpath = Path(wav_to_16000(wav_filepath))
                wav = wav_to_16000(wav_filepath)
                wav = preprocess_wav(fpath_or_wav=wav)

                encoder = VoiceEncoder()
                return encoder.embed_utterance(wav)

            return np.mean(cosine_similarity(
                get_speaker_ebd(synth_wav_path).reshape(1, -1),
                get_speaker_ebd(ref_wav_path).reshape(1, -1)
            ))
        
        speaker_cos = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing Speaker Cosine Similarity for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                speaker_cos_value = get_speaker_cos(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                speaker_cos.append(speaker_cos_value)

        return np.mean(np.array(speaker_cos))
    
    def compute_speaker_cos_direct(self):
        def get_speaker_cos(synth_wav_path, ref_wav_path):
            def get_speaker_ebd(wav_filepath):
                fpath = Path(wav_filepath)
                wav = preprocess_wav(fpath)

                encoder = VoiceEncoder()
                return encoder.embed_utterance(wav)

            return np.mean(cosine_similarity(
                get_speaker_ebd(synth_wav_path).reshape(1, -1),
                get_speaker_ebd(ref_wav_path).reshape(1, -1)
            ))

        speaker_cos = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing Speaker Cosine Similarity for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                speaker_cos_value = get_speaker_cos(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                speaker_cos.append(speaker_cos_value)
        return np.mean(np.array(speaker_cos))
    
    def compute_mfcc_cos(self):
        def get_pair_mfcc_cos(synth_wav_path, ref_wav_path):
            mfcc_synth = librosa.feature.mfcc(y=librosa.load(synth_wav_path)[0], sr=self.sampling_rate)
            mfcc_ref = librosa.feature.mfcc(y=librosa.load(ref_wav_path)[0], sr=self.sampling_rate)
            # Use fastdtw to align the two MFCC feature matrices
            _, path = fastdtw(mfcc_synth.T, mfcc_ref.T)
            # Aligned feature matrices
            aligned_mfcc_synth = mfcc_synth[:, [p[0] for p in path]].T
            aligned_mfcc_ref = mfcc_ref[:, [p[1] for p in path]].T
            # Normalize the aligned MFCC features
            aligned_mfcc_synth = aligned_mfcc_synth / np.linalg.norm(aligned_mfcc_synth, axis=0)
            aligned_mfcc_ref = aligned_mfcc_ref / np.linalg.norm(aligned_mfcc_ref, axis=0)
            return cosine_similarity(
                aligned_mfcc_synth.reshape(1, -1),
                aligned_mfcc_ref.reshape(1, -1)
            )
        mfcc_cos_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing MFCC Cosine Similarity for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                mfcc_cos_value = get_pair_mfcc_cos(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                mfcc_cos_list.append(mfcc_cos_value)
        mfcc_cos = np.mean(np.array(mfcc_cos_list)) if len(mfcc_cos_list) > 0 else float("-inf")
        return mfcc_cos
    
    def compute_mel_sdr(self):
        def calculate_sdr(synth_wav_path, ref_wav_path):

            # def compute_mel(wav_filepath):
            #     """
            #     This should not implement caching here, but due to environmental issues, it is implemented here.
            #     :param wav_filepath:
            #     :return:
            #     """

            #     def get_cache_filepath():
            #         wav_dir = osp.dirname(wav_filepath)
            #         cache_dir = osp.join(wav_dir + "_mel")
            #         base_name = osp.basename(wav_filepath) + ".npy"
            #         os.makedirs(cache_dir, exist_ok=True)
            #         return osp.join(cache_dir, base_name)

            #     cache_filepath = get_cache_filepath()
            #     if osp.exists(cache_filepath):
            #         mel_spectrogram = np.load(cache_filepath)
            #     else:
            #         raise NotImplementedError
            #     return mel_spectrogram  # (seq_len,80)
            
            def compute_mel(wav_filepath):
                """
                This should not implement the cache method here, but due to environmental issues, it is implemented here.
                :param wav_filepath:
                :return:
                """

                audio, sr = ta.load(wav_filepath)
                mel = mel_spectrogram(
                    y=audio, 
                    n_fft=1024, 
                    num_mels=80, 
                    sampling_rate=sr, 
                    hop_size=256,
                    win_size=1024,
                    fmin=0, 
                    fmax=8000, 
                    center=False
                ).squeeze()
                
                mel = mel.numpy() #[80, T]

                return mel.T # (T,80)

            eps = np.finfo(float).eps  # a very small value to ensure it's not zero
            f_synth = compute_mel(synth_wav_path).T  # (feature_dim,seq_len)
            f_ref = compute_mel(ref_wav_path).T  # (feature_dim,seq_len)
            # Use fastdtw to align the two MFCC feature matrices
            _, path = fastdtw(f_synth.T, f_ref.T)
            # Aligned feature matrices
            aligned_f_synth = f_synth[:, [p[0] for p in path]].T
            aligned_f_ref = f_ref[:, [p[1] for p in path]].T

            # Ensure both signals have the same length

            original = aligned_f_synth
            distorted = aligned_f_ref

            # Calculate the distortion signal
            distortion = distorted - original + eps

            # Calculate SDR
            sdr = 10 * np.log10(np.sum(original ** 2) / np.sum(distortion ** 2))

            return sdr

        mel_sdr_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing Mel SDR for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                mel_sdr_value = calculate_sdr(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                mel_sdr_list.append(mel_sdr_value)

        avg_mel_sdr = np.mean(np.array(mel_sdr_list)) if len(mel_sdr_list) > 0 else float("-inf")
        return avg_mel_sdr
    
    def compute_mfcc_e_cos(self):
        def get_pair_mfcc_cos(synth_wav_path, ref_wav_path):
            mfcc_synth = librosa.feature.mfcc(y=librosa.load(synth_wav_path)[0], sr=self.sampling_rate)
            mfcc_ref = librosa.feature.mfcc(y=librosa.load(ref_wav_path)[0], sr=self.sampling_rate)
            # Use fastdtw to align the two MFCC feature matrices
            _, path = fastdtw(mfcc_synth.T, mfcc_ref.T)
            # Aligned feature matrices
            aligned_mfcc_synth = mfcc_synth[:, [p[0] for p in path]].T
            aligned_mfcc_ref = mfcc_ref[:, [p[1] for p in path]].T  # (seq_len,20)
            # Normalize the aligned MFCC features
            aligned_mfcc_synth = aligned_mfcc_synth / np.linalg.norm(aligned_mfcc_synth, axis=0)
            aligned_mfcc_ref = aligned_mfcc_ref / np.linalg.norm(aligned_mfcc_ref, axis=0)
            cos_list = list()
            for i in range(len(aligned_mfcc_synth)):
                cos_list.append(cosine_similarity(
                    aligned_mfcc_synth[i].reshape(1, -1),
                    aligned_mfcc_ref[i].reshape(1, -1)
                ))
            return np.mean(np.array(cos_list))

        mfcc_e_cos_list = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing MFCC E-Cosine Similarity for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                mfcc_e_cos_value = get_pair_mfcc_cos(synth_wav_path=synth_wav_path, ref_wav_path=ref_wav_path)
                mfcc_e_cos_list.append(mfcc_e_cos_value)
        mfcc_e_cos = np.mean(np.array(mfcc_e_cos_list)) if len(mfcc_e_cos_list) > 0 else float("-inf")
        return mfcc_e_cos
    
    def compute_deep_speaker_cos(self):
        # SAMPLING_RATE = 22050
        trim_top_db = 23
        filter_length = 1024
        hop_length = 256
        arg_dit = {"sampling_rate": self.sampling_rate,
                   "win_length": 1024,
                   "speaker_embedder": "DeepSpeaker",
                   "speaker_embedder_cuda": False}

        def load_audio(wav_path):
            wav_raw, _ = librosa.load(wav_path, sr=self.sampling_rate)
            _, index = librosa.effects.trim(wav_raw, top_db=trim_top_db, frame_length=filter_length,
                                            hop_length=hop_length)
            wav = wav_raw[index[0]:index[1]]
            duration = (index[1] - index[0]) / hop_length
            return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

        args = argparse.Namespace(**arg_dit)
        speaker_emb = PreDefinedEmbedder(
            sampling_rate=args.sampling_rate,
            win_length=args.win_length,
            embedder_type=args.speaker_embedder,
            embedder_cuda=args.speaker_embedder_cuda,
        )
        cosine_score = []
        for speaker_id, wav_file_pairs in self.speakers_to_synth_wavs_and_reference.items():
            for wav_file_pair in tqdm(wav_file_pairs, desc=f"Computing Deep Speaker Cosine Similarity for speaker {speaker_id}"):
                synth_wav_path = wav_file_pair["synthesized_wav"]
                ref_wav_path = wav_file_pair["reference_wav"]
                score = 0.0
                synth_wav_raw, synth_wav, synth_duration = load_audio(synth_wav_path)
                synth_spker_embed = speaker_emb(synth_wav)
                ref_wav_raw, ref_wav, ref_duration = load_audio(ref_wav_path)
                ref_spker_embed = speaker_emb(ref_wav)
                score = cosine_similarity(ref_spker_embed, synth_spker_embed)
                cosine_score.append(score)
        return np.mean(np.array(cosine_score))
    
    def get_all_metrics(self):
        results = dict()
        # results["wer"] = self.compute_wer()
        # results["wer_un_comma"] = self.compute_wer_un_comma()
        # results["si_sdr"] = self.compute_si_sdr()
        # results["f0_corr"] = self.compute_f0_corr()
        # results["f0_rmse"] = self.compute_f0_rmse()
        # results["log_f0_rmse"] = self.compute_log_f0()
        results["ssim"] = self.compute_ssim()
        results["mcd"] = self.compute_mcd()
        results["mcd24"] = self.compute_mcd24()
        results["precision_mel"] = self.compute_precision_mel()
        results["precision_mfcc"] = self.compute_precision_mfcc()
        results["recall_mel"] = self.compute_recall_mel()
        results["recall_mfcc"] = self.compute_recall_mfcc()
        results["fid_mel"] = self.compute_fid_mel()
        results["fid_mfcc"] = self.compute_fid_mfcc()
        # results["fid_mfcc_un_norm"] = self.compute_fid_mfcc_un_norm()
        # results["mb_mos"] = self.compute_mb_mos()
        # results["ld_mos"] = self.compute_ld_mos()
        # results["mb_ld_mos"] = self.compute_mb_ld_mos()
        # results["target_mb_mos"] = self.compute_target_mb_mos()
        # results["target_ld_mos"] = self.compute_target_ld_mos()
        # results["target_mb_ld_mos"] = self.compute_target_mb_ld_mos()
        results["ffe"] = self.compute_ffe()
        #results["speaker_cos"] = self.compute_speaker_cos()
        results["mfcc_cos"] = self.compute_mfcc_cos()
        # results["mel_sdr"] = self.compute_mel_sdr()
        results["mfcc_e_cos"] = self.compute_mfcc_e_cos()
        # results["deep_speaker_cos"] = self.compute_deep_speaker_cos()
        return results
    
    def get_metrics_by_list(self, metric_list: List[str]):
        results = dict()
        for metric_name in metric_list:
            if not hasattr(self, f"compute_{metric_name}"):
                raise NotImplementedError(f"Metric {metric_name} is not implemented.")
            compute_func = getattr(self, f"compute_{metric_name}")
            results[metric_name] = compute_func()
        return results
