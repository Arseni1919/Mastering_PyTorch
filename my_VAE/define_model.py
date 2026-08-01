import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    per_sample_kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1)
    out = torch.mean(per_sample_kl, dim=0)
    return out


class Encoder(nn.Module):
    """
    encoder(x): # x: (bs, channels, side, side)
      h = conv stack, stride-2 each stage, doubling channels, with a nonlinearity between stages
      h = flatten(h) # (bs, flat_dim)
      mu = linear(h) # (bs, latent_dim) — no nonlinearity on this output
      logvar = linear(h) # (bs, latent_dim) — no nonlinearity on this output
      return mu, logvar
    """
    def __init__(self, channels, side_size, base_channels, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2)
        )
        flat_dim = 2 * base_channels * ((side_size // 4)**2)
        self.mu_layer = nn.Linear(flat_dim, latent_dim)
        self.logvar_layer = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        bs, ch, height, width = x.shape
        x = self.conv(x)
        x = x.reshape((bs, -1))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    decoder(z): # z: (bs, latent_dim)
      h = linear(z) # -> flat_dim
      h = reshape to the encoder's final conv shape (channels, side/2^num_stages, side/2^num_stages)
      h = conv stack, upsampling by 2 each stage, halving channels, mirroring the encoder's schedule in reverse
      return final_conv(h) # -> channels, with Tanh (matches [-1,1]-normalized input)
    """
    def __init__(self, channels, side_size, base_channels, latent_dim):
        super().__init__()
        self.channels = channels
        self.side_size = side_size
        self.base_channels = base_channels
        flat_dim = 2 * base_channels * ((side_size // 4) ** 2)
        self.linear = nn.Linear(latent_dim, flat_dim)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(2 * base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        bs, latent_size = z.shape
        h = self.linear(z)
        h = h.reshape((bs, 2 * self.base_channels, self.side_size // 4, self.side_size // 4))
        out = self.conv(h)
        return out


class VAEFullModel(nn.Module):
    def __init__(self, channels, side_size, base_channels, latent_dim):
        super().__init__()
        self.encoder = Encoder(channels, side_size, base_channels, latent_dim)
        self.decoder = Decoder(channels, side_size, base_channels, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def main():
    pass


if __name__ == '__main__':
    main()


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         pass
