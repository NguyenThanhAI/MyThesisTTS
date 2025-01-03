import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

from model.base import BaseModule, LayerNorm
from model.utils import pad, sequence_mask, fix_len_compatibility


class VariancePredictor(BaseModule):
    def __init__(self, 
                 in_channels: int, 
                 filter_channels: int, 
                 kernel_size: int, 
                 p_dropout: float):
        super(VariancePredictor, self).__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p=p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels=in_channels, 
                                      out_channels=filter_channels, 
                                      kernel_size=kernel_size, 
                                      padding=kernel_size//2)
        self.norm_1 = LayerNorm(channels=filter_channels)
        self.conv_2 = torch.nn.Conv1d(in_channels=filter_channels, 
                                      out_channels=filter_channels, 
                                      kernel_size=kernel_size, 
                                      padding=kernel_size//2)
        self.norm_2 = LayerNorm(channels=filter_channels)
        self.proj = torch.nn.Conv1d(in_channels=filter_channels, 
                                    out_channels=1, 
                                    kernel_size=1)
        

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(input=x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(input=x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)

        return x * x_mask


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len=None):
        output = list()
        mel_len = list()
        sum_duration = torch.sum(duration, dim=1)
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[1])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        sum_predicted = torch.sum(predicted)
        batch = batch.permute(1, 0)
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        out = out.permute(1, 0)
        return out

    def forward(self, x, duration, max_len=None):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
    

class VarianceAdaptor(BaseModule):
    def __init__(self,
                 stats_file_path: str,
                 in_channels: int=256,
                 filter_channels: int=256,
                 kernel_size: int=3,
                 p_dropout: float=0.2,
                 n_bins: int=256,
                 pitch_feature_level: str="phoneme_level",
                 energy_feature_level: str="phomeme_level",
                 pitch_quantization: str="log",
                 energy_quantization: str="log",
                 variance_dims: int=256):
        
        super(VarianceAdaptor, self).__init__()

        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        self.pitch_quantization = pitch_quantization
        self.energy_quantization = energy_quantization

        assert self.pitch_quantization in ["linear", "log"]
        assert self.energy_quantization in ["linear", "log"]

        with open(stats_file_path, "r") as f:
            stats = json.load(f)
            pitch_min = stats["pitch"]["min"]
            pitch_max = stats["pitch"]["max"]
            energy_min = stats["energy"]["min"]
            energy_max = stats["energy"]["max"]

        self.n_bins = n_bins
        self.variance_dims = variance_dims

        if self.pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                data=torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )

        else:
            self.pitch_bins = nn.Parameter(
                data=torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )

        if self.energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                data=torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                data=torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(num_embeddings=self.n_bins,
                                            embedding_dim=self.variance_dims)
        
        self.energy_embedding = nn.Embedding(num_embeddings=self.n_bins,
                                             embedding_dim=self.variance_dims)

        self.duration_predictor = VariancePredictor(in_channels=in_channels,
                                                    filter_channels=filter_channels,
                                                    kernel_size=kernel_size,
                                                    p_dropout=p_dropout)
        
        self.pitch_predictor = VariancePredictor(in_channels=in_channels,
                                                 filter_channels=filter_channels,
                                                 kernel_size=kernel_size,
                                                 p_dropout=p_dropout)
        
        self.energy_predictor = VariancePredictor(in_channels=in_channels,
                                                 filter_channels=filter_channels,
                                                 kernel_size=kernel_size,
                                                 p_dropout=p_dropout)
        
        self.length_regulator = LengthRegulator()

    def get_pitch_embedding(self, 
                            x: torch.Tensor,
                            x_mask: torch.Tensor,
                            target: torch.Tensor,
                            control: float):
        
        pitch_prediction = self.pitch_predictor(x=x, x_mask=x_mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(input=target, boundaries=self.pitch_bins))
        else:
            pitch_prediction = pitch_prediction * control
            embedding = self.pitch_embedding(torch.bucketize(input=pitch_prediction, boundaries=self.pitch_bins))

        embedding = embedding.squeeze(1).transpose(1, 2)

        return pitch_prediction, embedding
    
    def get_energy_embedding(self, 
                             x: torch.Tensor,
                             x_mask: torch.Tensor,
                             target: torch.Tensor,
                             control: float):
        
        energy_prediction = self.energy_predictor(x=x, x_mask=x_mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(input=target, boundaries=self.energy_bins))
        else:
            energy_prediction = energy_prediction * control
            embedding = self.energy_embedding(torch.bucketize(input=energy_prediction, boundaries=self.energy_bins))

        embedding = embedding.squeeze(1).transpose(1, 2)


        return energy_prediction, embedding
    
    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                y_max_length: torch.Tensor=None,
                duration_target: torch.Tensor=None,
                pitch_target: torch.Tensor=None,
                energy_target: torch.Tensor=None,
                d_control: float=1.0,
                p_control: float=1.0,
                e_control: float=1.0):
        
        # x_dp = x.detach()
        input_variance_adaptor = x.detach() + 0.1 * (x - x.detach())
        log_duration_prediction = self.duration_predictor(x=input_variance_adaptor, x_mask=x_mask)
        duration_rounded = torch.clamp((torch.ceil(torch.exp(log_duration_prediction)) * d_control), min=1)
        duration_rounded = duration_rounded.squeeze(1)
        if self.pitch_feature_level == "phoneme_level":
            # x_dp = x.detach()
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(input_variance_adaptor,#x=x, 
                                                                         x_mask=x_mask,
                                                                         target=pitch_target, 
                                                                         control=p_control)
            x = x + pitch_embedding

        if self.energy_feature_level == "phoneme_level":
            # x_dp = x.detach()
            energy_prediction, energy_embedding = self.get_energy_embedding(input_variance_adaptor,#x=x, 
                                                                            x_mask=x_mask,
                                                                            target=energy_target, 
                                                                            control=e_control)
            x = x + energy_embedding

        if y_max_length is not None:
            sum_duration_target = torch.sum(duration_target, dim=1)
            y_mask = sequence_mask(length=sum_duration_target, max_length=y_max_length)
            x, y_lengths = self.length_regulator(x=x, 
                                                 duration=duration_target, 
                                                 max_len=y_max_length)
            
        else:
            duration_sum = torch.sum(duration_rounded, dim=1).long()
            y_max_length = int(duration_sum.max())
            y_max_length_ = fix_len_compatibility(length=y_max_length)
            x, y_lengths = self.length_regulator(x=x, 
                                                 duration=duration_rounded,
                                                 max_len=y_max_length_)
            y_mask = sequence_mask(length=y_lengths, max_length=y_max_length_)

        

        y_mask = y_mask.unsqueeze(1)
        input_variance_adaptor = x.detach() + 0.1 * (x - x.detach())
        if self.pitch_feature_level == "frame_level":
            # x_dp = x.detach()
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(input_variance_adaptor,#x=x, 
                                                                         x_mask=y_mask,
                                                                         target=pitch_target, 
                                                                         control=p_control)
            x = x + pitch_embedding

        if self.energy_feature_level == "frame_level":
            # x_dp = x.detach()
            energy_prediction, energy_embedding = self.get_energy_embedding(input_variance_adaptor,#x=x,
                                                                            x_mask=y_mask,
                                                                            target=energy_target,
                                                                            control=e_control)
            
            x = x + energy_embedding

        return x, pitch_prediction, energy_prediction, log_duration_prediction, duration_rounded, y_lengths

# if __name__ == "__main__":

#     batch_size = 4
#     num_features = 256
#     filter_channels = 256


#     duration_predictor = VariancePredictor(in_channels=num_features,
#                                            filter_channels=filter_channels,
#                                            kernel_size=3,
#                                            p_dropout=0.1)

    
#     max_length = 200
#     x = torch.rand((batch_size, num_features, max_length), dtype=torch.float32)
#     lengths = torch.randint(0, max_length, size=[batch_size])
#     # print(lengths)
#     x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)

#     predicted_duration = duration_predictor(x=x, x_mask=x_mask)
#     print(predicted_duration.shape)
