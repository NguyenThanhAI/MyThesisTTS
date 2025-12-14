import librosa
import numpy as np
import scipy
import torch
import yaml

from metrics.mb_model import MBNet
from .ld_model.LDNet import LDNet


class MOSCal:
    def __init__(self, sample_rate=22500):
        self.sample_rate = sample_rate
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.ld_net_model = None
        self.mb_net_model = None

    def _load_mb_model(self):
        mb_model_path = "metrics/model-50000.pt"

        mb_net_model = MBNet(num_judges=5000).to(self.device)
        mb_net_model.load_state_dict(torch.load(mb_model_path, map_location="cpu"))
        mb_net_model.eval()
        return mb_net_model

    def _load_ld_model(self):
        ld_model_path = "metrics/Pretrained-LDNet-ML-2337/model-27000.pt"
        ld_config_path = "metrics/Pretrained-LDNet-ML-2337/config.yml"
        with open(ld_config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ld_net_model = LDNet(config).to(self.device)
        ld_net_model.load_state_dict(torch.load(ld_model_path, map_location="cpu"), strict=False)
        ld_net_model.eval()
        return ld_net_model

    def get_ld_mos(self, wav_path):
        if self.ld_net_model is None:
            self.ld_net_model = self._load_ld_model()
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, )
        wav = torch.tensor(
            np.abs(librosa.stft(wav, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)).T)
        wav = wav.to(self.device).unsqueeze(0)

        pred_mean_scores, posterior_scores = self.ld_net_model.average_inference(
            spectrum=wav,
            include_meanspk=False
        )
        return pred_mean_scores.detach().numpy()[0]

    def get_mb_mos(self, wav_path):
        if self.mb_net_model is None:
            self.mb_net_model = self._load_mb_model()
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, )
        wav = torch.tensor(
            np.abs(librosa.stft(wav, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)).T)
        wav = wav.to(self.device)
        # print(wav.size())
        wav = wav.unsqueeze(0).unsqueeze(1)
        mean_scores = self.mb_net_model.get_mean_mos(
            spectrum=wav,
        )
        # Predict for each frame, then average
        return torch.mean(mean_scores).detach().numpy()