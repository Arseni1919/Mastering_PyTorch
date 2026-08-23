import math
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from get_data import create_data


class Block(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class MLP(nn.Module):
    def __init__(self, channels_data: int = 2, layers: int = 5, channels: int = 512, channels_t: int = 512):
        super().__init__()
        self.channels_t = channels_t
        self.in_proj = nn.Linear(channels_data, channels)
        self.t_proj = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[
            Block(channels) for _ in range(layers)
        ])
        self.out_proj = nn.Linear(channels, channels_data)

    def get_t_embedding(self, t: torch.Tensor, max_positions: int = 10000):
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, t):
        t_emb = self.get_t_embedding(t)
        x = self.in_proj(x)
        t = self.t_proj(t_emb)
        x = x + t
        x = self.blocks(x)
        x = self.out_proj(x)
        return x


def main():
    net = MLP()
    x_list = []
    out_list = []
    samples = create_data()
    for sample in samples:
        sample = torch.tensor(sample).unsqueeze(0)
        t = torch.rand((1,))
        out = net(sample, t)
        x_list.append(sample)
        out_list.append(out)

    fig, ax = plt.subplots(1, 2)
    x = [i[0, 0].item() for i in out_list]
    y = [i[0, 1].item() for i in out_list]
    ax[0].scatter(x, y)
    x = [i[0, 0].item() for i in x_list]
    y = [i[0, 1].item() for i in x_list]
    ax[1].scatter(x, y)
    plt.show()


if __name__ == '__main__':
    main()