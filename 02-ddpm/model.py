import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_abar(T=1000, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
