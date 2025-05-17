import librosa
import parselmouth
import numpy as np
import torch
import torch.nn.functional as F
from pycwt import wavelet
from scipy.interpolate import interp1d

from typing import List, Union

gamma = 0
mcep_input = 3
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0: Union[np.ndarray, torch.Tensor]):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int32)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def norm_f0(f0: Union[np.ndarray, torch.Tensor], uv: Union[np.ndarray, torch.Tensor], f0_mean: float=0, f0_std: float=1, use_uv: bool=True, pitch_norm: str="log", pitch_norm_eps: float=1e-9):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == "standard":
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == "log":
        eps = pitch_norm_eps
        f0 = torch.log2(f0 + eps) if is_torch else np.log2(f0 + eps)
    if uv is not None and use_uv:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0: Union[np.ndarray, torch.Tensor], uv: Union[np.ndarray, torch.Tensor], f0_mean: float=0, f0_std: float=1, use_uv: bool=True, pitch_norm: str="log", pitch_norm_eps: float=1e-9):
    uv = f0 == 0
    f0 = norm_f0(f0=f0, uv=uv, 
                 f0_mean=f0_mean,
                 f0_std=f0_std,
                 use_uv=use_uv,
                 pitch_norm=pitch_norm,
                 pitch_norm_eps=pitch_norm_eps)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

    return f0, uv


def denorm_0(f0: Union[np.ndarray, torch.Tensor], uv: Union[np.ndarray, torch.Tensor], f0_mean: float=0, f0_std: float=1, use_uv: bool=True, pitch_norm: str="log", pitch_padding=None, min=None, max=None):
    if pitch_norm == "standard":
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == "log":
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and use_uv:
        min_len = np.min(np.array((f0.size(1), uv.size(1))))
        f0[:, :min_len][uv[:, :min_len] > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def get_pitch(wav_data: Union[np.ndarray, torch.Tensor],
              mel: Union[np.ndarray, torch.Tensor],
              sampling_rate: int,
              hop_length: int):
    time_step = hop_length / sampling_rate * 1000
    f0_min = 80
    f0_max = 750

    if hop_length == 128:
        pad_size = 4
    elif hop_length  == 256:
        pad_size = 2
    else:
        assert False

    f0 = parselmouth.Sound(wav_data, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]

    f0 = f0[:len(mel)-8] # to avoid negative rpad
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")
    # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
    # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
    # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0=f0)
    return f0, pitch_coarse
    

def expand_f0_ph(f0: Union[np.ndarray, torch.Tensor],
                 mel2ph: np.ndarray,
                 uv: Union[np.ndarray, torch.Tensor],
                 f0_mean: float=0, 
                 f0_std: float=1, 
                 use_uv: bool=True, 
                 pitch_norm: str="log", 
                 pitch_padding=None, 
                 min=None, 
                 max=None):
    f0 = denorm_0(f0=f0,
                  uv=uv,
                  f0_mean=f0_mean,
                  f0_std=f0_std,
                  use_uv=use_uv,
                  pitch_norm=pitch_norm,
                  pitch_padding=pitch_padding,
                  min=min,
                  max=max)

    f0 = F.pad(f0, [1, 0])
    f0 = torch.gather(f0, 1, mel2ph) # [B, T_mel]


#########
# cwt
#########


def load_wav(wave_file: str, sr: int):
    wav, _ = librosa.load(path=wave_file, sr=sr, mono=True)
    return wav


def convert_continuos_f0(f0: Union[np.ndarray, torch.Tensor]):
    """CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    f0 = np.copy(f0)
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("| all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_cont_lf0(f0: Union[np.ndarray, torch.Tensor]):
    uv, cont_f0_lpf = convert_continuos_f0(f0=f0)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf


def get_lf0_cwt(lf0: Union[np.ndarray, torch.Tensor]):
    """
    Input:
        signal of shape [N]
    Output:
        wavelet_lf0 of shape [10, N], scales of shape [10]
    """
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9
    wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    wavelet_lf0 = np.real(wavelet_lf0).T
    return wavelet_lf0, scales


def norm_scale(wavelet_lf0: Union[np.ndarray, torch.Tensor]):
    wavelet_lf0_norm = np.zeros((wavelet_lf0.shape[0], wavelet_lf0.shape[1]))
    mean = wavelet_lf0.mean(0)[None, :]
    std = wavelet_lf0.std(0)[None, :]
    wavelet_lf0_norm = (wavelet_lf0 - mean) / std
    return wavelet_lf0_norm, mean, std


def normalize_cwt_lf0(f0: Union[np.ndarray, torch.Tensor], mean: Union[np.ndarray, torch.Tensor], std: Union[np.ndarray, torch.Tensor]):
    uv, conf_lf0_lpf = get_cont_lf0(f0=f0)
    cont_lf0_norm = (conf_lf0_lpf - mean) / std
    wavelet_lf0, scales = get_lf0_cwt(lf0=cont_lf0_norm)
    wavelet_lf0_norm, _, _ = norm_scale(wavelet_lf0=wavelet_lf0)

    return wavelet_lf0_norm


def get_lf0_cwt_norm(f0s: List[Union[np.ndarray, torch.Tensor]], mean: Union[np.ndarray, torch.Tensor], std: Union[np.ndarray, torch.Tensor]):
    uvs: List = []
    cont_lf0_lpfs: List = []
    cont_lf0_lpf_norms: List = []
    wavelet_lf0s: List = []
    wavelet_lf0s_norm: List = []
    scaless: List = []

    means: List = []
    stds: List = []

    for f0 in f0s:
        uv, cont_lf0_lpf = get_cont_lf0(f0=f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std

        wavelet_lf0, scales = get_lf0_cwt(lf0=cont_lf0_lpf_norm) # [560, 10]
        wavelet_lf0_norm, mean_scale, std_scale = norm_scale(wavelet_lf0=wavelet_lf0) # [560, 10], [1, 10], [1, 10]

        wavelet_lf0s_norm.append(wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        wavelet_lf0s.append(wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return wavelet_lf0s_norm, scaless, means, stds


def invert_cwt_torch(wavelet_lf0: torch.Tensor, scales: List[torch.Tensor]):
    b = ((torch.arange(0, len(scales)).float().to(wavelet_lf0.device)[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    return lf0_rec_sum


def inverse_cwt(wavelet_f0: np.ndarray, scales: List[np.ndarray]):
    b = ((np.arange(0, len(scales))[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = wavelet_f0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdims=True)) / lf0_rec_sum.std(-1, keepdims=True)
    
    return lf0_rec_sum


def cwt2f0(cwt_spec: Union[np.ndarray, torch.Tensor], mean: Union[np.ndarray, torch.Tensor], std: Union[np.ndarray, torch.Tensor], cwt_scales: Union[np.ndarray, torch.Tensor]):
    assert len(mean.shape) == 1 and len(std.shape) == 1 and len(cwt_spec.shape) == 3

    if isinstance(cwt_spec, torch.Tensor):
        f0 = invert_cwt_torch(wavelet_lf0=cwt_spec, scales=cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = f0.exp() # [B, T]
    else:
        f0 = inverse_cwt(wavelet_f0=cwt_spec, scales=cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = np.exp(f0) # [B, T]

    return f0


def cwt2f0_norm(cwt_spec: Union[np.ndarray, torch.Tensor], mean: Union[np.ndarray, torch.Tensor], std: Union[np.ndarray, torch.Tensor], mel2ph: Union[np.ndarray, torch.Tensor], cwt_scales: Union[np.ndarray, torch.Tensor],
                uv: Union[np.ndarray, torch.Tensor], f0_mean: float=0, f0_std: float=1, use_uv: bool=True, pitch_norm: str="log", pitch_norm_eps: float=1e-9):
    f0 = cwt2f0(cwt_spec=cwt_spec, mean=mean, std=std, cwt_scales=cwt_scales)
    f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), dim=1)
    f0_norm = norm_f0(f0, uv=uv, f0_mean=f0_mean, f0_std=f0_std,
                      use_uv=use_uv, pitch_norm=pitch_norm, pitch_norm_eps=pitch_norm_eps)

    return f0_norm