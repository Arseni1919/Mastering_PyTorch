from typing import List, Tuple

import torch
import torch.nn as nn
from config import config


def build_2d_rope(num_patches_per_side, head_dim):
    """
    """
    quater = head_dim // 4
    arrange_list = torch.arange(num_patches_per_side)
    # step 1
    row_idx, col_idx = torch.meshgrid(arrange_list, arrange_list, indexing='ij')
    row_idx = row_idx.reshape(-1)
    col_idx = col_idx.reshape(-1)
    # step 2
    freq_bands = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(quater) / quater)
    # step 3
    row_angles = torch.outer(row_idx, freq_bands)
    col_angles = torch.outer(col_idx, freq_bands)
    # step 4
    half_angles = torch.cat([row_angles, col_angles], dim=-1)
    # step 5
    angles = torch.cat([half_angles, half_angles], dim=-1)
    # step 6
    return torch.cos(angles), torch.sin(angles)


def modulate(x, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d1, d2 = cos.shape
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.reshape((1,1,d1,d2)) + rotated * sin.reshape((1,1,d1,d2))


def unpatchify(tokens: torch.Tensor, patch_size: int, channels: int, side: int):
    """
    unpatchify(tokens):  # tokens: (bs, num_patches, patch_size*patch_size*channels)
      reshape each token back into its patch_size x patch_size x channels patch
      reassemble the grid of patches into the full (bs, channels, side, side) image
    """
    bs, num_patches, hidden_dim = tokens.shape
    num_patches_per_side = side // patch_size
    tokens = tokens.reshape((bs, num_patches_per_side, num_patches_per_side, patch_size, patch_size, channels))
    tokens = tokens.permute(0, 5, 1, 3, 2, 4)
    tokens = tokens.reshape((bs, channels, side, side))
    return tokens


class Patchify(nn.Module):
    """
    patchify(x):  # x: (bs, ch, side, side)
      split into non-overlapping patch_size x patch_size patches, project each to hidden_size
      # standard trick: a single conv with kernel_size=stride=patch_size does patch-split + projection in one op
      return tokens: (bs, num_patches, hidden_size)     # num_patches = (side/patch_size)^2
    """
    def __init__(self, patch_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )


    def forward(self, x):
        out = self.conv(x)
        bs, ch, wd, hg = out.shape
        out = out.permute(0, 2, 3, 1).reshape(bs, wd*hg, ch)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_embedding = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

    def sin_cos_emb(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(half) / half)
        freqs = freqs.to(t.device)
        args = torch.unsqueeze(t * 1000, -1) * freqs
        t_emb = torch.concat([args.sin(), args.cos()], dim=-1)
        return t_emb

    def forward(self, t):
        t_emb = self.sin_cos_emb(t)
        t_emb = self.time_embedding(t_emb)
        return t_emb


class AdaLN(nn.Module):
    """
    adaLN(cond):  # cond: (bs, hidden_size)
      out = MLP(SiLU(cond))  # -> 6 * hidden_size
      split into 6 chunks, each (bs, hidden_size):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
      return these 6 vectors
    """
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6*hidden_size)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x) -> Tuple[torch.Tensor]:
        out = self.mlp(x).chunk(6, dim=-1)
        return out


class MultiHeadAttnBlock(nn.Module):
    """
    attn(x, rope_cos, rope_sin):  # x: (bs, num_patches, hidden_size)
      q, k, v = linear projections of x, each (bs, num_patches, hidden_size)
      reshape each to (bs, num_heads, num_patches, head_dim)     # head_dim = hidden_size / num_heads
      q = apply_rope(q, rope_cos, rope_sin)
      k = apply_rope(k, rope_cos, rope_sin)  # v is NOT rotated — only q/k need positional info for the dot product
      attn_weights = softmax(q @ k^T / sqrt(head_dim))  # over num_patches
      out = attn_weights @ v
      merge heads back to (bs, num_patches, hidden_size)
      return output_projection(out)
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        bs, num_patches, hidden_size = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        q = q.reshape((bs, num_patches, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)
        k = k.reshape((bs, num_patches, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)
        v = v.reshape((bs, num_patches, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        out = attn_weights @ v
        out = out.permute(0, 2, 1, 3).reshape((bs, num_patches, hidden_size))
        out = self.out_proj(out)
        return out


class DiTBlock(nn.Module):
    """
    dit_block(x, cond, rope_cos, rope_sin):
      shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaLN(cond)
      h = modulate(layernorm(x), shift_attn, scale_attn)
      x = x + gate_attn.unsqueeze(1) * attn(h, rope_cos, rope_sin)
      h = modulate(layernorm(x), shift_mlp, scale_mlp)
      x = x + gate_mlp.unsqueeze(1) * mlp(h)  # mlp: Linear(hidden, hidden*mlp_ratio) -> GELU -> Linear(back to hidden)
      return x
    """
    def __init__(self, cond_dim, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.adaln = AdaLN(cond_dim=cond_dim, hidden_size=hidden_size)
        self.attn = MultiHeadAttnBlock(hidden_size, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size)
        )

    def forward(self, x, cond, rope_cos, rope_sin):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp  = self.adaln(cond)
        h = modulate(self.layer_norm(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.attn(h, rope_cos, rope_sin)
        h = modulate(self.layer_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """
    final_layer(x, cond):
      shift, scale = a smaller adaLN variant of cond -> 2 * hidden_size, split in two
      x = modulate(layernorm(x), shift, scale)
      return linear(x)  # hidden_size -> patch_size*patch_size*channels
    """
    def __init__(self, cond_dim, hidden_size, patch_size, channels):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.channels = channels
        self.mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*channels)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond) -> Tuple[torch.Tensor]:
        shift, scale = self.mlp(cond).chunk(2, dim=-1)
        x = modulate(self.layer_norm(x), shift, scale)
        return self.linear(x)


class DiTFullModel(nn.Module):
    """
    # in __init__: rope_cos, rope_sin = build_2d_rope(side/patch_size, head_dim)   # computed once, reused every forward call

    dit_forward(x, t, class_labels):
      tokens = patchify(x)                     # no positional embeddings added here anymore — RoPE lives inside attention
      cond = time_embed(t) + class_embed(class_labels)

      for block in blocks:                    # num_layers identical DiT blocks
        tokens = block(tokens, cond, rope_cos, rope_sin)

      tokens = final_layer(tokens, cond)
      return unpatchify(tokens)                # same shape as input x
    """
    def __init__(
            self,
            side,
            path_side,
            head_dim,
            hidden_size,
            num_classes,
            class_emb_size,
            time_emb_size,
            num_layers,
            num_heads,
            mlp_ratio,
            channels
    ):
        super().__init__()
        self.side = side
        self.path_side = path_side
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.class_emb_size = class_emb_size
        self.time_emb_size = time_emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.channels = channels
        rope_cos, rope_sin = build_2d_rope(side//path_side, head_dim)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)
        self.patchify = Patchify(path_side, hidden_size)
        self.time_embedding = TimeEmbedding(time_emb_size)
        self.class_embedding = nn.Embedding(num_classes, class_emb_size)
        cond_dim = time_emb_size + class_emb_size
        self.blocks = nn.ModuleList([
            DiTBlock(cond_dim, hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer(cond_dim, hidden_size, path_side, channels)

    def forward(self, x, t, class_labels):
        tokens = self.patchify(x)
        cond = torch.cat([self.time_embedding(t), self.class_embedding(class_labels)], dim=-1)
        for block in self.blocks:
            tokens = block(tokens, cond, self.rope_cos, self.rope_sin)
        tokens = self.final_layer(tokens, cond)
        out = unpatchify(tokens, self.path_side, self.channels, self.side)
        return out





def check():
    x = torch.rand((3, 1, 32, 32))
    t = torch.rand(3, 1)
    class_labels = torch.randint(0, 10, (3, 1))
    net: DiTFullModel = DiTFullModel(
        side=config.side_size,
        path_side=config.patch_size,
        head_dim=config.hidden_size // config.num_heads,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        class_emb_size=config.class_emb_size,
        time_emb_size=config.time_emb_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        channels=config.channels
    )
    out = net(x, t, class_labels)
    print(out.shape)
    # patch_layer = Patchify(patch_size=config.patch_size, hidden_size=config.hidden_size)
    # out = patch_layer(x)
    # a, b = build_2d_rope(2, 8)
    # print(a.shape)
    # print(f'{config.patch_size=}')
    # print(f'{config.hidden_size=}')
    # print(out.shape)

if __name__ == '__main__':
    check()


# class AdaLN(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         pass