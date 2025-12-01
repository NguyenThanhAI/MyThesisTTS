# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule
from model.base import ConvNorm
import torch.nn.functional as F


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(input=torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=dim, 
                                             out_channels=dim, 
                                             kernel_size=4, 
                                             stride=2, 
                                             padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=dim, 
                                    out_channels=dim, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(data=torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(in_channels=dim, 
                                                         out_channels=dim_out, 
                                                         kernel_size=3, 
                                                         padding=1), 
                                        torch.nn.GroupNorm(num_groups=groups, num_channels=dim_out), 
                                        Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), 
                                       torch.nn.Linear(in_features=time_emb_dim, 
                                                       out_features=dim_out))

        self.block1 = Block(dim=dim, 
                            dim_out=dim_out, 
                            groups=groups)
        self.block2 = Block(dim=dim_out, 
                            dim_out=dim_out, 
                            groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(in_channels=dim, 
                                            out_channels=dim_out, 
                                            kernel_size=1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(in_channels=dim, 
                                      out_channels=hidden_dim * 3, 
                                      kernel_size=1, 
                                      bias=False)
        self.to_out = torch.nn.Conv2d(in_channels=hidden_dim, 
                                      out_channels=dim, 
                                      kernel_size=1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / self.dim
        emb = torch.exp(2 * torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        # emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # return emb
        pe = torch.zeros(x.shape[0], self.dim, device=device)
        pe[:, ::2] = emb.sin()
        pe[:, 1::2] = emb.cos()
        return pe


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(in_features=spk_emb_dim, out_features=spk_emb_dim * 4), 
                                               Mish(),
                                               torch.nn.Linear(in_features=spk_emb_dim * 4, out_features=n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim=dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=dim, out_features=dim * 4), 
                                       Mish(),
                                       torch.nn.Linear(in_features=dim * 4, out_features=dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=dim),
                       ResnetBlock(dim=dim_out, dim_out=dim_out, time_emb_dim=dim),
                       Residual(fn=Rezero(fn=LinearAttention(dim=dim_out))),
                       Downsample(dim=dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(fn=Rezero(fn=LinearAttention(dim=mid_dim)))
        self.mid_block2 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim=dim_out * 2, dim_out=dim_in, time_emb_dim=dim),
                     ResnetBlock(dim=dim_in, dim_out=dim_in, time_emb_dim=dim),
                     Residual(fn=Rezero(fn=LinearAttention(dim=dim_in))),
                     Upsample(dim=dim_in)]))
        self.final_block = Block(dim=dim, dim_out=dim)
        self.final_conv = torch.nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        
        self.estimator = GradLogPEstimator2d(dim=dim, n_spks=n_spks,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(t=time, beta_init=self.beta_min, beta_term=self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(t=time, beta_init=self.beta_min, beta_term=self.beta_max, cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z=z, mask=mask, mu=mu, n_timesteps=n_timesteps, stoc=stoc, spk=spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0=x0, mask=mask, mu=mu, t=t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(t=time, beta_init=self.beta_min, beta_term=self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0=x0, mask=mask, mu=mu, t=t, spk=spk)


class DenoiserTrigFlow(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(DenoiserTrigFlow, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(in_features=spk_emb_dim, out_features=spk_emb_dim * 4), 
                                               Mish(),
                                               torch.nn.Linear(in_features=spk_emb_dim * 4, out_features=n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim=dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=dim, out_features=dim * 4), 
                                       Mish(),
                                       torch.nn.Linear(in_features=dim * 4, out_features=dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=dim),
                       ResnetBlock(dim=dim_out, dim_out=dim_out, time_emb_dim=dim),
                       Residual(fn=Rezero(fn=LinearAttention(dim=dim_out))),
                       Downsample(dim=dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(fn=Rezero(fn=LinearAttention(dim=mid_dim)))
        self.mid_block2 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim=dim_out * 2, dim_out=dim_in, time_emb_dim=dim),
                     ResnetBlock(dim=dim_in, dim_out=dim_in, time_emb_dim=dim),
                     Residual(fn=Rezero(fn=LinearAttention(dim=dim_in))),
                     Upsample(dim=dim_in)]))
        self.final_block = Block(dim=dim, dim_out=dim)
        self.final_conv = torch.nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)

        self.logvar_linear = torch.nn.Linear(in_features=dim, out_features=1)

    def forward(self, x, mask, mu, t, spk=None, return_logvar=True):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        logvar = self.logvar_linear(t)

        if return_logvar:
            return (output * mask).squeeze(1), logvar
        else:
            return (output * mask).squeeze(1)

from functools import partial
import numpy as np

class ConsitencyTrigFlow(BaseModule):
    def __init__(
            self, 
            n_feats, 
            dim,
            num_warmup_steps: int,
            n_spks=1, 
            spk_emb_dim=64,
            pe_scale=1000,
            ema_rate: float=0.98):
        super(ConsitencyTrigFlow, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.num_warmup_steps = num_warmup_steps
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        self.ema_rate = ema_rate

        self.estimator = DenoiserTrigFlow(
            dim=dim,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            pe_scale=pe_scale
        )

        # self.target_estimator = DenoiserTrigFlow(
        #     dim=dim,
        #     n_spks=n_spks,
        #     spk_emb_dim=spk_emb_dim,
        #     pe_scale=pe_scale
        # )
        # self.target_estimator.requires_grad_(False)
        # self.copy_target_params()

        self.sigma_data: float=0.5

    # def copy_target_params(self):
    #     for param, target_param in zip(
    #         self.estimator.parameters(), 
    #         self.target_estimator.parameters()
    #     ):
    #         target_param.data.copy_(param.data)

    # def update_ema_target_params(self):
    #     for param, target_param in zip(
    #         self.estimator.parameters(), 
    #         self.target_estimator.parameters()
    #     ):
    #         target_param.data.mul_(self.ema_rate).add_(
    #             param.data, alpha=1 - self.ema_rate
    #         )

    # def update_ema_target_params(self):
    #     with torch.no_grad():
    #         for param, target_param in zip(
    #             self.estimator.parameters(),
    #             self.target_estimator.parameters()
    #         ):
    #             target_param.mul_(self.ema_rate).add_(param, alpha=1 - self.ema_rate)

    def forward_diffusion(self, x0, mask, t):
        time = t.unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False) * self.sigma_data # [batch_size, 80, seq_len]
        x_t = torch.cos(time) * x0 + torch.sin(time) * z # [batch_size, 80, seq_len]

        return x_t * mask, z * mask
    
    @staticmethod
    def calc_karras_sigmas(sigma_min=0.002, sigma_max=80.0, steps=50, rho=7):
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    
    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, spk=None):
        sampling_timesteps = torch.arctan(self.calc_karras_sigmas(sigma_min=0.002, sigma_max=80, steps=n_timesteps, rho=7) / self.sigma_data).to(z.device)

        start_t = sampling_timesteps[0]
        # start_t = start_t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        start_t = start_t.unsqueeze(-1).unsqueeze(-1)
        x_t = z * mask * self.sigma_data
        x = torch.cos(start_t) * x_t - torch.sin(start_t) * self.sigma_data * self.estimator.forward(x=x_t / self.sigma_data, mask=mask, mu=mu, t=start_t.flatten(), spk=spk, return_logvar=False)

        for t in sampling_timesteps[1:]:
            noise = torch.randn(
                z.shape, dtype=z.dtype, device=z.device, requires_grad=False
            ) * self.sigma_data
            t = t.unsqueeze(-1).unsqueeze(-1)
            x_t = torch.cos(t) * x + torch.sin(t) * noise
            x = torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * self.estimator.forward(x=x_t / self.sigma_data, mask=mask, mu=mu, t=t.flatten(), spk=spk, return_logvar=False)

        x = x * mask
        return x
    
    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, spk=None):
        return self.reverse_diffusion(z=z, mask=mask, mu=mu, n_timesteps=n_timesteps, spk=spk)
    
    def sample_t(self, x0):
        p_mean = -0.8
        p_std = 1.6

        t = torch.randn(x0.shape[0], device=x0.device, requires_grad=False) # [batch_size]
        t = (p_mean + p_std * t).exp() # [batch_size]
        t = torch.arctan(t / self.sigma_data) # [batch_size]

        return t
    
    @staticmethod
    def model_wrapper(scaled_x_t, t, estimator, mask, mu, spk):
        pred, logvar = estimator.forward(
            x=scaled_x_t,
            mask=mask,
            mu=mu,
            t=t.flatten(),
            spk=spk
        )

        return pred, logvar
    
    def loss_t(self, x0, mask, mu, t, step, spk=None):
        x_t, z = self.forward_diffusion(x0=x0, mask=mask, t=t)
        time = t.unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
        dxt_dt = torch.cos(time) * z - torch.sin(time) * x0 # [batch_size, 1, 1]

        v_x = torch.cos(time) * torch.sin(time) * dxt_dt / self.sigma_data
        v_t = torch.cos(time) * torch.sin(time)

        model_wrapper_partial = partial(
            self.model_wrapper,
            estimator=self.estimator,
            mask=mask,
            mu=mu,
            spk=spk
        )

        F_theta, F_theta_grad, logvar = torch.func.jvp(
            model_wrapper_partial,
            (x_t / self.sigma_data, time),
            (v_x, v_t),
            has_aux=True
        )
        logvar = logvar.view(-1, 1, 1)
        F_theta_grad = F_theta_grad.detach()
        F_theta_minus = F_theta.detach()

        # F_theta, logvar = self.estimator.forward(
        #     x=x_t / self.sigma_data,
        #     mask=mask,
        #     mu=mu,
        #     t=time.flatten(),
        #     spk=spk
        # )

        # logvar = logvar.view(-1, 1, 1)

        r = min(1.0, step / self.num_warmup_steps)

        # Calculate gradient g using JVP rearrangement
        g = -torch.cos(time) * torch.cos(time) * (self.sigma_data * F_theta_minus - dxt_dt)
        
        second_term = -r * torch.cos(time) * torch.sin(time) * (x_t + self.sigma_data * F_theta_grad)

        g = g + second_term

        # Tangent normalization
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2), keepdim=True)
        
        g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())

        g = g / (g_norm + 0.1)


        # weight = 1
        prior_weight = 1 / (self.sigma_data * torch.tan(time))
        # loss = (weight / (torch.exp(logvar) * x0[0].numel())) * torch.square(F_theta - F_theta_minus - g).sum(dim=(1, 2), keepdim=True) + logvar
        # loss = (weight / (torch.exp(logvar))) * torch.square(F_theta - F_theta_minus - g) + logvar
        loss = (torch.exp(logvar) * prior_weight / x0[0].numel()) * torch.square(F_theta - F_theta_minus - g).sum(dim=(1, 2), keepdim=True) - logvar
        loss = loss.mean()
        # loss_g = (weight / torch.exp(logvar)) * torch.square(g) + logvar
        # loss_g = loss_g.mean()

        return loss, x_t
    
    def compute_loss(self, x0, mask, mu, step, spk=None):
        t = self.sample_t(x0=x0)
        return self.loss_t(
            x0=x0,
            mask=mask,
            mu=mu,
            t=t,
            step=step,
            spk=spk,
        )
    

class LinearNorm(BaseModule):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            torch.nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class ResNetBlockDenoiser(BaseModule):
    def __init__(self, decoder_dim, speaker_emb_dim, n_feats):
        super(ResNetBlockDenoiser, self).__init__()
        self.decoder_dim = decoder_dim
        self.speaker_emb_dim = speaker_emb_dim
        self.n_feats = n_feats

        self.speaker_projection = LinearNorm(
            in_features=speaker_emb_dim, 
            out_features=decoder_dim,
            bias=True
        )

        self.condition_projection = ConvNorm(
            in_channels=n_feats, 
            out_channels=decoder_dim, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        self.time_projection = LinearNorm(
            in_features=decoder_dim, 
            out_features=decoder_dim,
            bias=True
        )

        self.conv_layer = ConvNorm(
            in_channels=decoder_dim,
            out_channels=decoder_dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.output_projection = ConvNorm(
            in_channels=decoder_dim,
            out_channels=decoder_dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, cond, time_emb, speaker_emb, mask):
        """
        x: (B, n_feats, T) - noisy mel-spectrogram
        t: (B, decoder_dim) - diffusion step
        cond: (B, n_feats, T) - conditional features
        speaker_emb: (B, speaker_emb_dim) - speaker embedding
        mask: (B, 1, T) - padding mask
        Returns:
        """
        # Time embedding
        # ttime_emb = self.time_pos_emb(t)  # (B, decoder_dim)
        # time_emb = self.time_mlp(time_emb)  # (B, decoder_dim)
        t_emb = self.time_projection(time_emb)  # (B, decoder_dim)

        # Speaker embedding projection
        speaker_emb = self.speaker_projection(speaker_emb)  # (B, decoder_dim)

        # Condition projection
        cond = self.condition_projection(cond, mask)  # (B, decoder_dim, T)

        # Combine all embeddings
        # x: (B, C, T)
        # t_emb: (B, decoder_dim) -> (B, decoder_dim, 1)
        # speaker_emb: (B, decoder_dim) -> (B, decoder_dim, 1)
        residual = x + t_emb.unsqueeze(-1)
        x = residual + speaker_emb.unsqueeze(-1) + cond

        x = self.conv_layer(x, mask)  # (B, decoder_dim * 2, T)

        gate, filter = torch.chunk(x, 2, dim=1)  # (B, decoder_dim, T) each
        x = torch.sigmoid(gate) * torch.tanh(filter)  # (B, decoder_dim, T)

        y = self.output_projection(x, mask)  # (B, decoder_dim * 2, T)
        x, skip = torch.chunk(y, 2, dim=1)  # (B, decoder_dim, T) each

        return (x + residual) / math.sqrt(2.0), skip
    

class ConsistencyDenoiser(BaseModule):
    def __init__(
            self, 
            num_blocks, 
            decoder_dim, 
            speaker_emb_dim, 
            n_feats,
            pe_scale: int=10
        ):
        super(ConsistencyDenoiser, self).__init__()
        self.num_blocks = num_blocks
        self.decoder_dim = decoder_dim
        self.speaker_emb_dim = speaker_emb_dim
        self.n_feats = n_feats
        self.pe_scale = pe_scale

        self.time_pos_emb = SinusoidalPosEmb(dim=decoder_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=decoder_dim, out_features=decoder_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=decoder_dim * 4, out_features=decoder_dim),
            torch.nn.SiLU(),
        )

        self.input_projection = ConvNorm(
            in_channels=n_feats,
            out_channels=decoder_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            )

        self.resnet_blocks = torch.nn.ModuleList(
            [
                ResNetBlockDenoiser(
                    decoder_dim=decoder_dim,
                    speaker_emb_dim=speaker_emb_dim,
                    n_feats=n_feats,
                )
                for _ in range(num_blocks)
            ]
        )

        self.skip_projection = ConvNorm(
            in_channels=decoder_dim,
            out_channels=decoder_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.output_projection = ConvNorm(
            in_channels=decoder_dim,
            out_channels=n_feats,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, cond, t, spk, mask):
        """
        x: (B, n_feats, T) - noisy mel-spectrogram
        cond: (B, n_feats, T) - conditional features
        t: (B, ) - diffusion step
        spk: (B, speaker_emb_dim) - speaker embedding
        mask: (B, 1, T) - padding mask
        Returns:
        """
        time_emb = self.time_pos_emb.forward(t, scale=self.pe_scale)  # (B, decoder_dim)
        time_emb = self.time_mlp(time_emb) # (B, decoder_dim)
        x = self.input_projection(x, mask)  # (B, decoder_dim, T)
        x = F.relu(x)
        skip_connections = []
        for resnet_block in self.resnet_blocks:
            x, skip = resnet_block(
                x, cond, time_emb, spk, mask
            )  # x: (B, decoder_dim, T), skip: (B, decoder_dim, T)
            skip_connections.append(skip)

        x = torch.sum(torch.stack(skip_connections, dim=0), dim=0) / math.sqrt(self.num_blocks)  # (B, decoder_dim, T)
        x = self.skip_projection(x, mask)  # (B, decoder_dim, T)
        x = F.relu(x)
        x = self.output_projection(x, mask)  # (B, n_feats, T)

        return x
    

class ConsistencyDiffusion(BaseModule):
    def __init__(
            self, 
            n_feats, 
            dim,
            num_warmup_steps: int,
            total_steps: int,
            num_blocks: int=20,
            spk_emb_dim=64,
            pe_scale=1,
            ema_rate: float=0.98,
            sigma_max: float=80.0,
            sigma_min: float=0.002,
            rho: float=7.0,
            sigma_data: float=0.5,
            start_scales: int=2,
            end_scales: int=200,
            weight_schedule: str="karras"
        ):
        super(ConsistencyDiffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.num_warmup_steps = num_warmup_steps
        self.num_blocks = num_blocks
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        self.ema_rate = ema_rate

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.sigma_data: float=sigma_data
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.num_scales = self.start_scales
        self.weight_schedule = weight_schedule

        self.estimator = ConsistencyDenoiser(
            num_blocks=num_blocks,
            decoder_dim=dim,
            speaker_emb_dim=spk_emb_dim,
            n_feats=n_feats,
            pe_scale=pe_scale
        )

        self.target_estimator = ConsistencyDenoiser(
            num_blocks=num_blocks,
            decoder_dim=dim,
            speaker_emb_dim=spk_emb_dim,
            n_feats=n_feats,
            pe_scale=pe_scale
        )

        self.target_estimator.requires_grad_(False)
        self.copy_target_params()

        self.ema_and_scales_fn = self.create_ema_and_scales_fn(
            start_ema=self.ema_rate,
            start_scales=self.start_scales,
            end_scales=self.end_scales,
            total_steps=total_steps
        )
    
    def copy_target_params(self):
        for param, target_param in zip(
            self.estimator.parameters(), 
            self.target_estimator.parameters()
        ):
            target_param.data.copy_(param.data)


    def update_ema_target_params(self):
        with torch.no_grad():
            for param, target_param in zip(
                self.estimator.parameters(),
                self.target_estimator.parameters()
            ):
                target_param.mul_(self.ema_rate).add_(param, alpha=1 - self.ema_rate)

    def create_ema_and_scales_fn(
        self,
        start_ema,
        start_scales,  # 2
        end_scales,
        total_steps,
    ):
        def ema_and_scales_fn(step):
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
            return float(target_ema), int(scales)
        
        return ema_and_scales_fn

    def compute_t_from_indices(self, indices):
        t = self.sigma_max ** (1 / self.rho) + indices / (self.num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t ** self.rho
        return t

    def forward_diffusion(self, x0, mask, indices, z):
        # z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
        #                 requires_grad=False)
        t = self.compute_t_from_indices(indices)
        time = t.unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
        x_t = x0 + time * z

        return x_t * mask, t
    
    @staticmethod
    def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.to(device)
    
    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, spk=None):
        x_t = z * mask
        # sigmas = self.get_sigmas_karras(
        #     n=n_timesteps,
        #     sigma_min=self.sigma_min,
        #     sigma_max=self.sigma_max,
        #     rho=self.rho,
        #     device=z.device
        # )
        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        for i in range(n_timesteps - 1):
            t = (t_max_rho + i / (n_timesteps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            t = torch.tensor([t], device=z.device)
            pred_x0 = self.denoiser_wrapper(
                x_t=x_t,
                t=t,
                mask=mask,
                mu=mu,
                spk=spk
            )
            next_t = (t_max_rho + (i + 1) / (n_timesteps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = torch.tensor(next_t, device=z.device)
            x_t = pred_x0 + torch.rand_like(z) * torch.sqrt(next_t**2 - self.sigma_min**2)

        return pred_x0

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, spk=None):
        return self.reverse_diffusion(z=z, mask=mask, mu=mu, n_timesteps=n_timesteps, spk=spk)
    
    def get_snr(self, sigmas):
        return sigmas ** -2
    
    def get_weightings(self, snrs):
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data ** 2
        elif self.weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()
        return weightings
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
    
    def get_scalings_for_boundary_condition(self, sigma):
        """

        :param sigma:
        :return:
        """
        c_skip = self.sigma_data ** 2 / (
                (sigma - self.sigma_min) ** 2 + self.sigma_data ** 2
        )
        c_out = (
                (sigma - self.sigma_min)
                * self.sigma_data
                / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        )
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    
    def denoiser_wrapper(self, x_t, t, mask, mu, spk):
        
        c_skip, c_out, c_in = self.get_scalings_for_boundary_condition(t)
        rescaled_t = 1000 * 0.25 * torch.log(t + 1e-12)
        pred = self.estimator.forward(
            x=c_in.unsqueeze(-1).unsqueeze(-1) * x_t,
            cond=mu,
            t=rescaled_t,
            spk=spk,
            mask=mask,
        )
        return c_skip.unsqueeze(-1).unsqueeze(-1) * x_t + c_out.unsqueeze(-1).unsqueeze(-1) * pred
    
    @staticmethod
    def masked_smooth_l2_with_weight(x, y, mask, weights, c=0.07):
        # x, y:    [B, 80, T]
        # mask:    [B, 1, T]  (0/1 float)
        # weights: [B]

        # Mở rộng mask từ [B, 1, T] -> [B, 80, T]
        mask_expanded = mask.expand_as(x)

        # Áp mask vào diff
        diff = (x - y) * mask_expanded                 # [B, 80, T]

        # Tính L2^2 theo từng sample
        sq = (diff ** 2).sum(dim=(1, 2))               # [B]

        # Đếm số frame hợp lệ theo từng sample
        num_valid = mask_expanded.sum(dim=(1, 2)) + 1e-9        # [B], tránh chia 0

        # Chuẩn hóa theo số frame
        sq = sq / num_valid                            # [B]

        # Smooth L2 loss
        loss = torch.sqrt(sq + c*c) - c                # [B]

        # Nhân weights per sample
        loss = loss * weights
        return loss                       # [B]
    
    def loss_t(self, x0, mask, mu, indices, step, spk=None):
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                         requires_grad=False)
        x_t, t = self.forward_diffusion(x0=x0, mask=mask, indices=indices, z=z)
        x_t2, t2 = self.forward_diffusion(x0=x0, mask=mask, indices=indices + 1, z=z)

        snrs = self.get_snr(t)
        # if weights is None:
        weights = self.get_weightings(snrs)

        denoised_pred = self.denoiser_wrapper(
            x_t=x_t,
            t=t,
            mask=mask,
            mu=mu,
            spk=spk
        )

        denoised_target = self.denoiser_wrapper(
            x_t=x_t2,
            t=t2,
            mask=mask,
            mu=mu,
            spk=spk
        )

        denoised_target = denoised_target.detach()

        diff_loss = self.masked_smooth_l2_with_weight(
            x=denoised_pred,
            y=denoised_target,
            mask=mask,
            weights=weights
        )
        diff_loss = diff_loss.mean()

        recon_loss = self.masked_smooth_l2_with_weight(
            x=denoised_pred,
            y=x0,
            mask=mask,
            weights=weights
        )
        recon_loss = recon_loss.mean()

        return diff_loss, recon_loss, x_t
    

    def compute_loss(self, x0, mask, mu, step, spk=None):
        target_ema, num_scales = self.ema_and_scales_fn(step)
        self.ema_rate = target_ema
        if self.num_scales != num_scales:
            print(f"Updating num_scales: {self.num_scales} -> {num_scales}")
        self.num_scales = num_scales

        indices = torch.randint(
            low=0,
            high=self.num_scales - 1,
            size=(x0.shape[0],),
            device=x0.device,
            dtype=torch.int32,
            requires_grad=False
        )

        diff_loss, recon_loss, x_t = self.loss_t(
            x0=x0,
            mask=mask,
            mu=mu,
            indices=indices,
            step=step,
            spk=spk
        )

        return diff_loss, recon_loss, x_t