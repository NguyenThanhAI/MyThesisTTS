import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModule
from model.utils import sequence_mask


def mae_loss(target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor):

    mae = torch.abs(target - prediction)
    mae = mae * mask
    mae = torch.sum(mae)
    num_elements = torch.sum(mask)
    loss = mae / (num_elements * prediction.shape[1])

    return loss

def mse_loss(target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor):

    mse = (target - prediction) ** 2
    mse = mse * mask
    mse = torch.sum(mse)
    num_elements = torch.sum(mask)
    loss = mse / (num_elements * prediction.shape[1])

    return loss


class TotalLoss(BaseModule):

    def __init__(self, pitch_feature_level, energy_feature_level):
        super(TotalLoss, self).__init__()

        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, 
                x_lengths: torch.Tensor, y_lengths: torch.Tensor,
                mel_target: torch.Tensor,
                pitch_target: torch.Tensor, energy_target: torch.Tensor, duration_target: torch.Tensor,
                mel_prediction: torch.Tensor,
                pitch_prediction: torch.Tensor, energy_prediction: torch.Tensor, log_duration_prediction: torch.Tensor):
        
        x_mask = sequence_mask(x_lengths)
        x_mask = x_mask.unsqueeze(1)
        y_max_length = mel_target.shape[-1]
        y_mask = sequence_mask(y_lengths, max_length=y_max_length)
        y_mask = y_mask.unsqueeze(1)

        mel_target.requires_grad = False
        pitch_target.requires_grad = False
        energy_target.requires_grad = False
        duration_target.requires_grad = False

        mel_loss = mae_loss(target=mel_target, prediction=mel_prediction, mask=y_mask)

        if self.pitch_feature_level == "phoneme_level":
            pitch_loss = mse_loss(target=pitch_target, prediction=pitch_prediction, mask=x_mask)
        else:
            pitch_loss = mse_loss(target=pitch_target, prediction=pitch_prediction, mask=y_mask)

        if self.energy_feature_level == "phoneme_level":
            energy_loss = mse_loss(target=energy_target, prediction=energy_prediction, mask=x_mask)
        else:
            energy_loss = mse_loss(target=energy_target, prediction=energy_prediction, mask=y_mask)

        log_duration_target = torch.log(1e-8 + duration_target).unsqueeze(1)

        duration_loss = mse_loss(target=log_duration_target, prediction=log_duration_prediction, mask=x_mask)

        total_loss = mel_loss + pitch_loss + energy_loss + duration_loss

        return total_loss, mel_loss, pitch_loss, energy_loss, duration_loss