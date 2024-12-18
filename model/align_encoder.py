import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from beartype import beartype
from beartype.typing import Optional

from model.base import BaseModule
from model.utils import sequence_mask

def exists(val):
    return val is not None


class AlignerEncoder(BaseModule):
    
    def __init__(self,
                 in_dims: int=80,
                 hidden_dims: int=512,
                 attn_channels: int=80,
                 kernel_size: int=3,
                 temperature: float=5e-4):
        
        super(AlignerEncoder, self).__init__()

        self.temperature = temperature

        self.key_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dims,
                      out_channels=hidden_dims * 2,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=hidden_dims * 2,
                      out_channels=attn_channels,
                      kernel_size=1,
                      padding=1//2,
                      bias=True)
        ])

        self.query_layers = nn.ModuleList([
            nn.Conv1d(in_channels=in_dims,
                      out_channels=in_dims * 2,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_dims * 2, 
                      out_channels=in_dims,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_dims,
                      out_channels=attn_channels,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        ])

    @beartype
    def forward(self,
                queries: torch.Tensor, # Melspectrogram [B, C, T]
                keys: torch.Tensor, # Phonemes features [B, C, N]
                queries_mask: Optional[torch.Tensor]=None,
                keys_mask: Optional[torch.Tensor]=None):

        key_out = keys
        for layer in self.key_layers:
            if isinstance(layer, nn.Conv1d):
                key_out = key_out * keys_mask
            key_out = layer(key_out)

        key_out = key_out * keys_mask # [B, C, N]

        query_out = queries
        for layer in self.query_layers:
            if isinstance(layer, nn.Conv1d):
                query_out = query_out * queries_mask
            query_out = layer(query_out)

        query_out = query_out * queries_mask # [B, C, T]

        key_out = rearrange(key_out, "b c t -> b t c") # [B, C, N] -> [B, N, C]
        query_out = rearrange(query_out, "b c t -> b t c") # [B, C, T] -> [B, T, C]

        attn_logp = - torch.cdist(query_out, key_out) # [B, T, N]
        attn_logp = rearrange(attn_logp, "b ... -> b 1 ...") # [B, T, N] -> [B, 1, T, N]

        mask = queries_mask.unsqueeze(-1) * keys_mask.unsqueeze(2) # [B, 1, T, N]
        mask = mask.bool()
        attn_logp.data.masked_fill_(~mask, -torch.finfo(attn_logp.dtype).max)

        attn = attn_logp.softmax(dim=-1) # [B, 1, T, N]
        return attn, attn_logp


def pad_tensor(input, pad, value=0):
    pad = [item for sublist in reversed(pad) for item in sublist]  # Flatten the tuple
    assert len(pad) // 2 == len(input.shape), "Padding dimensions do not match input dimensions"
    return F.pad(input, pad, mode="constant", value=value)


def maximum_path(value, mask, const=None):
    device = value.device
    dtype = value.dtype
    if not exists(const):
        const = torch.tensor(float("-inf")).to(device)  # Patch for Sphinx complaint
    value = value * mask

    b, t_x, t_y = value.shape
    direction = torch.zeros(value.shape, dtype=torch.int64, device=device)
    v = torch.zeros((b, t_x), dtype=torch.float32, device=device)
    x_range = torch.arange(t_x, dtype=torch.float32, device=device).view(1, -1)

    for j in range(t_y):
        v0 = pad_tensor(v, ((0, 0), (1, 0)), value = const)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = torch.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = torch.where(index_mask.view(1,-1), v_max + value[:, :, j], const)

    direction = torch.where(mask.bool(), direction, 1)

    path = torch.zeros(value.shape, dtype=torch.float32, device=device)
    index = mask[:, :, 0].sum(1).long() - 1
    index_range = torch.arange(b, device=device)

    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1

    path = path * mask.float()
    path = path.to(dtype=dtype)
    return path



class ForwardSumLoss(BaseModule):
    def __init__(self, blank_logprob: float=-1) -> None:
        super().__init__()

        self.blank_logprob = blank_logprob
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(blank=0,
                                         zero_infinity=True)


    def forward(self, attn_logprob, phoneme_lens, mel_lens):
        """
        attn_logprob: [Batch_size, 1, mel_lens, phoneme_lens]
        phoneme_lens: [Batch_size]
        mel_lens: [Batch_size]
        """
        device, blank_logprob  = attn_logprob.device, self.blank_logprob

        # Add blank label
        attn_logprob_padded = F.pad(attn_logprob, (1, 0, 0, 0, 0, 0), value = blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, phoneme_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[:mel_lens[bid], :, :phoneme_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                log_probs=curr_logprob,
                targets=target_seq,
                input_lengths=mel_lens[bid : bid + 1],
                target_lengths=phoneme_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


class BinLoss(BaseModule):

    def __init__(self):
        super().__init__()

    def forward(self, alignment_hard, alignment_soft):
        """
        alignment_hard: torch.Tensor
            hard alignment map [B, mel_lens, phoneme_lens]
        alignment_soft: torch.Tensor
            soft alignment potentials [B, mel_lens, phoneme_lens]
        """
        log_sum = torch.log(
            torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)
        ).sum()
        return - log_sum / alignment_hard.sum()


class Aligner(BaseModule):
    def __init__(self,
                 in_dims: int=80,
                 hidden_dims: int=512,
                 attn_channels: int=80,
                 kernel_size: int=3,
                 temperature: float=5e-4):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.attn_channels = attn_channels
        self.kernel_size = kernel_size
        self.temperature = temperature

        self.aligner = AlignerEncoder(in_dims=self.in_dims,
                                      hidden_dims=self.hidden_dims,
                                      attn_channels=self.attn_channels,
                                      kernel_size=self.kernel_size,
                                      temperature=self.temperature)
        
    def forward(self, 
                x: torch.Tensor,
                x_mask: torch.Tensor,
                y: torch.Tensor,
                y_mask: torch.Tensor):
        
        alignment_soft, alignment_logprob = self.aligner(queries=y, 
                                                         keys=x, 
                                                         queries_mask=y_mask, 
                                                         keys_mask=x_mask)

        x_mask = rearrange(x_mask, "... i -> ... i 1") # [B, 1, N] -> [B, 1, N, 1]
        y_mask = rearrange(y_mask, "... j -> ... 1 j") # [B, 1, T] -> [B, 1, 1, T]
        attn_mask = x_mask * y_mask # [B, 1, N, T]
        attn_mask = rearrange(attn_mask, "b 1 i j -> b i j") # [B, 1, N, T] -> [B, N, T]

        alignment_soft = rearrange(alignment_soft, "b 1 c t -> b t c") # [B, 1, T, N] -> [B, T, N] -> [B, N, T]
        alignment_mask = maximum_path(alignment_soft, attn_mask) # [B, N, T]
        
        alignment_hard = torch.sum(alignment_mask, -1).int() # [B, N]
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mask
    


if __name__ == "__main__":
    batch_size = 10
    seq_len_y = 200   # length of sequence y
    seq_len_x = 35
    feature_dim = 80  # feature dimension

    x = torch.randn(batch_size, 512, seq_len_x)
    y = torch.randn(batch_size, seq_len_y, feature_dim)
    y = y.transpose(1,2) #dim-1 is the channels for conv
    
    # Create masks
    x_mask = torch.ones(batch_size, 1, seq_len_x)
    y_mask = torch.ones(batch_size, 1, seq_len_y)

    aligner = Aligner(in_dims=80,
                    hidden_dims=512,
                    attn_channels=80,
                    kernel_size=3,
                    temperature=5e-4)
    
    alignment_hard, alignment_soft, alignment_logprob, alignment_mask = aligner(x=x, x_mask=x_mask, y=y, y_mask=y_mask)