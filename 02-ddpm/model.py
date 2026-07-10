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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(2, 3)
        q, k, v = qkv.unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, ch=128, mults=(1, 2, 2, 2), n_classes=10, emb_dim=512, attn_res=16):
        super().__init__()
        self.ch = ch
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.label_emb = nn.Embedding(n_classes + 1, emb_dim)
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.down, skips, cur, res = nn.ModuleList(), [ch], ch, 32
        for i, m in enumerate(mults):
            for _ in range(2):
                stage = nn.ModuleList([ResBlock(cur, ch * m, emb_dim)])
                cur = ch * m
                if res == attn_res:
                    stage.append(Attention(cur))
                self.down.append(stage)
                skips.append(cur)
            if i < len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(cur)]))
                skips.append(cur)
                res //= 2
        self.mid = nn.ModuleList(
            [ResBlock(cur, cur, emb_dim), Attention(cur), ResBlock(cur, cur, emb_dim)]
        )
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(3):
                stage = nn.ModuleList([ResBlock(cur + skips.pop(), ch * m, emb_dim)])
                cur = ch * m
                if res == attn_res:
                    stage.append(Attention(cur))
                self.up.append(stage)
            if i > 0:
                self.up.append(nn.ModuleList([Upsample(cur)]))
                res *= 2
        self.norm_out = nn.GroupNorm(32, cur)
        self.conv_out = nn.Conv2d(cur, 3, 3, padding=1)

    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.ch)) + self.label_emb(y)
        h = self.conv_in(x)
        skips = [h]
        for stage in self.down:
            h = self.run_stage(stage, h, emb)
            skips.append(h)
        for block in self.mid:
            h = self.run_stage(nn.ModuleList([block]), h, emb)
        for stage in self.up:
            if isinstance(stage[0], Upsample):
                h = stage[0](h)
                continue
            h = self.run_stage(stage, torch.cat([h, skips.pop()], dim=1), emb)
        return self.conv_out(F.silu(self.norm_out(h)))

    def run_stage(self, stage, h, emb):
        for block in stage:
            h = block(h, emb) if isinstance(block, ResBlock) else block(h)
        return h
