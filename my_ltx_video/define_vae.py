from statistics import mean
from typing import Any

import torch.nn as nn
import torch

from config import config


def pixel_norm(x):
    return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


def unpatchify3d(h, vae_patch_size, channels):
    # in (bs, channels, num_frames, H/patch_size, W/patch_size)
    bs, in_channels, num_frames, in_h, in_w = h.shape
    h = h.reshape((bs, vae_patch_size, vae_patch_size, num_frames, in_h, in_w))
    # h = h.permute(0, 3, 1, 4, 2, 5)
    h = h.permute(0, 3, 4, 1, 5, 2)
    h = h.reshape((bs, channels, num_frames, vae_patch_size*in_h, vae_patch_size*in_w))
    # out (bs, channels, num_frames, H, W)
    return h


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    noise = torch.randn_like(mu)
    return mu + std * noise


def kl_divergence(mu, logvar):
    """
    kl_divergence(mu, logvar):
      the standard closed-form KL divergence between a Gaussian(mu, exp(logvar)) and the standard normal prior
      sum that across every non-batch dimension (channels, time, height, width) for each sample
      average the resulting per-sample values across the batch
    """
    return -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=(1,2,3,4)).mean()


class CasualConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size_t, self.kernel_size_w, self.kernel_size_h = kernel_size
        self.stride = stride
        if padding is None:
            pad_side = self.kernel_size_w // 2
            padding = (0, pad_side, pad_side)
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor):
        # bs, channels, num_frames, H, W = x.shape
        first_frame = x[:,:,:1,:,:]
        first_frames = first_frame.repeat((1,1,self.kernel_size_t - 1,1,1))
        x = torch.cat([first_frames, x], dim=2)
        x = self.conv3d(x)
        return x


class Patchify3D(nn.Module):
    def __init__(self, channels, vae_base_channels, vae_patch_size):
        super().__init__()
        self.channels = channels
        self.vae_base_channels = vae_base_channels
        self.vae_patch_size =  vae_patch_size
        kernel_size = (1,vae_patch_size, vae_patch_size)
        stride = (1, vae_patch_size, vae_patch_size)
        padding = (0,0,0)
        self.conv3d = CasualConv3D(
            in_channels=channels, out_channels=vae_base_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor):
        # x: (bs, channels, num_frames, H, W)
        x = self.conv3d(x)
        # return patched: (bs, vae_base_channels, num_frames, H/patch_size, W/patch_size)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.silu = nn.SiLU()
        self.casual_conv3d_1 = CasualConv3D(
            in_channels, out_channels,
            kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1)
        )
        self.casual_conv3d_2 = CasualConv3D(
            out_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        if self.in_channels != self.out_channels:
            self.casual_conv3d_3 = CasualConv3D(
                in_channels, out_channels,
                kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
            )

    def forward(self, h: torch.Tensor):
        bs, channels, T, H, W = h.shape
        skip = h
        h = pixel_norm(h)
        h = self.silu(h)
        h = self.casual_conv3d_1(h)
        h = pixel_norm(h)
        h = self.silu(h)
        h = self.casual_conv3d_2(h)
        if self.in_channels != self.out_channels:
            skip = self.casual_conv3d_3(skip)
        return h + skip


class Downsample(nn.Module):
    def __init__(self, in_channels, shrink_time=False):
        super().__init__()
        self.in_channels = in_channels
        self.shrink_t_rate = 2 if shrink_time else 1
        self.casual_conv3d = CasualConv3D(
            in_channels, in_channels,
            kernel_size=(self.shrink_t_rate, 2, 2), stride=(self.shrink_t_rate, 2, 2), padding=(0, 0, 0)
        )

    def forward(self, h: torch.Tensor):
        return self.casual_conv3d(h)


class Upsample(nn.Module):
    """
    upsample(h): nearest/bilinear upsample + a refine conv,
    same upsample-then-refine idea as my_ddpm's UpBlock
    """
    def __init__(self, in_channels, expand_time=False):
        super().__init__()
        self.in_channels = in_channels
        self.expand_t_rate = 2 if expand_time else 1
        self.upsample = nn.Upsample(scale_factor=(self.expand_t_rate, 2, 2), mode='nearest')
        self.casual_conv3d = CasualConv3D(
            in_channels, in_channels,
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, h: torch.Tensor):
        h = self.upsample(h)
        h = self.casual_conv3d(h)
        return h


class Encoder(nn.Module):
    def __init__(
            self,
            channels,
            vae_base_channels,
            vae_patch_size,
            num_downsample_stages,
            num_temporal_downsample_stages,
            vae_blocks_per_stage,
            vae_latent_channels
    ):
        super().__init__()
        self.channels = channels
        self.vae_base_channels = vae_base_channels
        self.vae_patch_size = vae_patch_size
        self.num_downsample_stages = num_downsample_stages
        self.num_temporal_downsample_stages = num_temporal_downsample_stages
        self.vae_blocks_per_stage = vae_blocks_per_stage
        self.vae_latent_channels = vae_latent_channels
        self.patchify3d = Patchify3D(channels, vae_base_channels, vae_patch_size)
        self.stages = nn.ModuleList()
        c = vae_base_channels
        for i_downsample_stage in range(num_downsample_stages-1):
            for i_block_per_stage in range(vae_blocks_per_stage):
                in_channels = c if i_block_per_stage == 0 else 2*c
                res_block = ResBlock(in_channels=in_channels, out_channels=2*c)
                self.stages.append(res_block)
            shrink_time = i_downsample_stage < num_temporal_downsample_stages
            downsample_layer = Downsample(in_channels=2*c, shrink_time=shrink_time)
            self.stages.append(downsample_layer)
            c = 2*c
        self.bottleneck = nn.ModuleList()
        for i_block_per_stage in range(vae_blocks_per_stage):
            res_block = ResBlock(in_channels=c, out_channels=c)
            self.bottleneck.append(res_block)
        self.conv_mu = CasualConv3D(c, vae_latent_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.conv_logvar = CasualConv3D(c, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x: torch.Tensor):
        """
        h = patchify(x)
        run h through each entry in stages, in order
        run h through each entry in bottleneck, in order
        return conv_mu(h), conv_logvar(h)
        """
        h = self.patchify3d(x)
        for layer in self.stages:
            h = layer(h)
        for layer in self.bottleneck:
            h = layer(h)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
            self,
            channels,
            vae_base_channels,
            vae_patch_size,
            num_downsample_stages,
            num_temporal_downsample_stages,
            vae_blocks_per_stage,
            vae_latent_channels
    ):
        super().__init__()
        self.channels = channels
        self.vae_base_channels = vae_base_channels
        self.vae_patch_size = vae_patch_size
        self.num_downsample_stages = num_downsample_stages
        self.num_temporal_downsample_stages = num_temporal_downsample_stages
        self.vae_blocks_per_stage = vae_blocks_per_stage
        self.vae_latent_channels = vae_latent_channels

        c = vae_base_channels*(2**(num_downsample_stages - 1))
        self.conv_in = CasualConv3D(vae_latent_channels, c, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.bottleneck = nn.ModuleList()
        for i_block_per_stage in range(vae_blocks_per_stage):
            res_block = ResBlock(c, c)
            self.bottleneck.append(res_block)
        self.stages = nn.ModuleList()
        for i_downsample_stage in range(num_downsample_stages - 1):
            matching_encoder_stage = (num_downsample_stages - 2) - i_downsample_stage
            expand_time = matching_encoder_stage < num_temporal_downsample_stages
            upsample = Upsample(c, expand_time=expand_time)
            self.stages.append(upsample)
            for i_block_per_stage in range(vae_blocks_per_stage):
                out_channels = c // 2 if i_block_per_stage == (vae_blocks_per_stage - 1) else c
                res_block = ResBlock(c, out_channels=out_channels)
                self.stages.append(res_block)
            c = c // 2
        self.conv_out = CasualConv3D(c, channels*vae_patch_size**2, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        for layer in self.bottleneck:
            h = layer(h)
        for layer in self.stages:
            h = layer(h)
        h = self.conv_out(h)
        h = torch.tanh(h)
        h = unpatchify3d(h, self.vae_patch_size, self.channels)
        return h


def main():
    # x = torch.rand(2, 1, 8, 32, 32)
    # encoder = Encoder(
    #     config.channels, config.vae_base_channels, config.vae_patch_size, config.num_downsample_stages,
    #     config.vae_blocks_per_stage, config.vae_latent_channels
    # )
    # mu, logvar = encoder(x)
    # print(f'{mu.shape=}')
    # print(f'{logvar.shape=}')
    # mu = torch.rand(2, 8, 8, 8, 8)
    # logvar = torch.rand(2, 1, 8, 8, 8)
    # z = reparameterize(mu, logvar)
    # print(z.shape)
    # z = torch.rand(2, 16, 8, 8, 8)
    # decoder = Decoder(
    #     config.channels, config.vae_base_channels, config.vae_patch_size, config.num_downsample_stages,
    #     config.vae_blocks_per_stage, config.vae_latent_channels
    # )
    # x_hat = decoder(z)
    # print(x_hat.shape)
    # expected: x_hat.shape == (2, 1, 8, 32, 32)
    mu = torch.zeros(2, 8, 8, 8, 8)
    logvar = torch.zeros(2, 1, 8, 8, 8)
    kl = kl_divergence(mu, logvar)
    print(kl)


if __name__ == '__main__':
    main()


# class ResBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor):
#         return x