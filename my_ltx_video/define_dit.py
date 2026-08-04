import torch
import torch.nn as nn

from config import config


def build_3d_rope(T, H, W, head_dim):
  sixth = head_dim // 6
  t_idx, h_idx, w_idx = torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing='ij')
  t_idx = t_idx.reshape(-1)
  h_idx = h_idx.reshape(-1)
  w_idx = w_idx.reshape(-1)

  freq_bands = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(sixth) / sixth)

  t_angles = torch.outer(t_idx, freq_bands)
  h_angles = torch.outer(h_idx, freq_bands)
  w_angles = torch.outer(w_idx, freq_bands)

  half_angles = torch.cat([t_angles, h_angles, w_angles], dim=-1)   # (num_tokens, head_dim // 2)
  angles = torch.cat([half_angles, half_angles], dim=-1)   # (num_tokens, head_dim)
  return torch.cos(angles), torch.sin(angles)


def sinusoidal_embedding(t: torch.Tensor, dim):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(half) / half).to(t.device)
    args = torch.unsqueeze(t, -1) * freqs
    emb = torch.concat([args.sin(), args.cos()], dim=-1)
    return emb


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d1, d2 = cos.shape
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.reshape((1,1,d1,d2)) + rotated * sin.reshape((1,1,d1,d2))


def modulate(x, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unflatten_tokens(tokens: torch.Tensor, T, H, W):
    bs, T_H_W, vae_latent_channels = tokens.shape
    z_hat = tokens.reshape((bs, T, H, W, vae_latent_channels)).permute(0, 4, 1, 2, 3)
    return z_hat



class TokenEmbed(nn.Module):
    def __init__(self, vae_latent_channels, hidden_size):
        super().__init__()
        self.linear = nn.Linear(vae_latent_channels, hidden_size)

    def forward(self, x: torch.Tensor):
        bs, vae_latent_channels, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape((bs, T*H*W, vae_latent_channels))
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable = nn.Parameter(torch.ones((dim,)))

    def forward(self, x: torch.Tensor):
        rms = (x**2 + 1e-9).mean(dim=-1, keepdim=True).sqrt()
        x = x / rms
        x = x * self.learnable
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.SiLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )

    def forward(self, t: torch.Tensor):
        t_emb = sinusoidal_embedding(t*1000, self.hidden_size)
        t_emb = self.mlp(t_emb)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor):
        out: torch.Tensor = self.mlp(cond)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = out.chunk(6, dim=-1)
        return shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rms_norm_q = RMSNorm(self.head_dim)
        self.rms_norm_k = RMSNorm(self.head_dim)


    def forward(self, x: torch.Tensor, rope_cos, rope_sin):
        bs, num_tokens, hidden_size = x.shape
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.reshape((bs, num_tokens, self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.reshape((bs, num_tokens, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape((bs, num_tokens, self.num_heads, self.head_dim)).transpose(1, 2)
        q = self.rms_norm_q(q)
        k = self.rms_norm_k(k)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape((bs, num_tokens, hidden_size))
        out = self.out_proj(out)
        return out


class DiTBlock(nn.Module):
    def __init__(self, cond_dim, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.adaln = AdaLN(cond_dim, hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear( mlp_ratio * hidden_size, hidden_size)
        )

    def forward(self, x: torch.Tensor, cond, rope_cos, rope_sin):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(cond)
        h = modulate(self.layer_norm(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.attn(h, rope_cos, rope_sin)
        h = modulate(self.layer_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, cond_dim, hidden_size, vae_latent_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, vae_latent_channels)

    def forward(self, x: torch.Tensor, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)
        x = modulate(self.layer_norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiTFullModel(nn.Module):
    def __init__(
            self,
            vae_latent_channels,
            hidden_size,
            num_heads,
            num_layers,
            mlp_ratio,
            num_classes,
            class_emb_size,
            T, H, W
    ):
        super().__init__()
        self.cond_dim = hidden_size + class_emb_size
        self.head_dim = hidden_size // num_heads
        self.token_embed = TokenEmbed(vae_latent_channels, hidden_size)
        self.time_embed = TimeEmbedding(hidden_size)
        self.class_embed = nn.Embedding(num_classes, class_emb_size)
        self.T, self.H, self.W = T, H, W
        rope_cos, rope_sin = build_3d_rope(T, H, W, head_dim=self.head_dim)
        self.register_buffer('rope_cos',  rope_cos)
        self.register_buffer('rope_sin',  rope_sin)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = DiTBlock(self.cond_dim, hidden_size, num_heads, mlp_ratio)
            self.blocks.append(block)
        self.final_layer = FinalLayer(self.cond_dim, hidden_size, vae_latent_channels)


    def forward(self, z, t, class_labels):
        tokens = self.token_embed(z)
        cond = torch.cat([self.time_embed(t), self.class_embed(class_labels)], dim=-1)
        for block in self.blocks:
            tokens = block(tokens, cond, self.rope_cos, self.rope_sin)
        tokens = self.final_layer(tokens, cond)
        out = unflatten_tokens(tokens, self.T, self.H, self.W)
        return out

def main():
    net = DiTFullModel(config.vae_latent_channels, config.hidden_size, config.num_heads, config.num_layers, config.mlp_ratio, config.num_classes, config.class_emb_size, 4, 8, 8)
    z = torch.rand(2, config.vae_latent_channels, 4, 8, 8)
    t = torch.rand(2)
    class_labels = torch.randint(0, config.num_classes, (2,))
    out = net(z, t, class_labels)
    print(out.shape)
    # expected: out.shape == z.shape — the DiT predicts a velocity field over the same latent grid it consumed
    # token_embed = TokenEmbed(16, 128)
    # z = torch.rand(2, 16, 4, 8, 8)
    # tokens = token_embed(z)
    # print(tokens.shape)
    # rms_norm = RMSNorm(8)
    # x = torch.rand(2, 4, 64, 8)  # (bs, num_heads, num_tokens, head_dim)
    # out = rms_norm(x)
    # print(x.shape)
    # print(out.shape)
    # hidden_size = 16
    # class_emb_size = 8
    # final_layer = FinalLayer(hidden_size + class_emb_size, hidden_size, 32)
    # x = torch.rand(2, 4 * 8 * 8, hidden_size)
    # cond = torch.rand(2, hidden_size + class_emb_size)
    # out = final_layer(x, cond)
    # print(out.shape) # expected: out.shape == (2, 4*8*8, vae_latent_channels)
if __name__ == '__main__':
    main()


# class TokenEmbed(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor):
#         return x