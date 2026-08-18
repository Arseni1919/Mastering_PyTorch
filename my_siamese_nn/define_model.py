import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.func import functional_call
from get_data import dataset


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.in_proj(x))
        for layer in self.layers:
            h = self.relu(layer(x))
            x = x + h
        x = self.out_proj(x)
        return x


class HyperNet(nn.Module):
    def __init__(self, input_net: nn.Module, in_dim: int, hidden_dim: int):
        super().__init__()
        self.out_dim = 0
        for name, p in input_net.named_parameters():
            self.out_dim += p.numel()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Switcher:
    def __init__(self, main_net, hyper_net):
        self.names = [name for name, _ in main_net.named_parameters()]
        self.numels = [p.numel() for _, p in main_net.named_parameters()]
        self.shapes = [p.shape for _, p in main_net.named_parameters()]
        self.num_params = sum(self.numels)
        self.main_net = main_net
        self.hyper_net = hyper_net

    def __call__(self, gray_image: torch.Tensor, coordinates: torch.Tensor):
        weights = self.hyper_net(gray_image)
        batch_size, _ = weights.shape
        params = {}
        outs = []
        for i in range(batch_size):
            chunks = weights[i].split(self.numels, dim=-1)
            for name, chunk, shape in zip(self.names, chunks, self.shapes):
                params[name] = chunk.reshape(shape)
            out = functional_call(self.main_net, params, coordinates[i])
            outs.append(out)
        outs = torch.stack(outs, dim=0)
        return outs


def main():
    x, y = dataset[random.randint(0, len(dataset) - 1)]

    grayscale_transform = torchvision.transforms.Grayscale()
    gray_image = grayscale_transform(x)
    ch, h_, w_ = gray_image.shape
    coordinates = torch.Tensor([0, 0])

    classifier = Classifier(in_dim=2, hidden_dim=128, out_dim=3, num_layers=10)
    hyper_net = HyperNet(input_net=classifier, in_dim=ch * h_ * w_, hidden_dim=128)
    switcher = Switcher(classifier, hyper_net)

    out = switcher(gray_image.flatten().unsqueeze(0), coordinates.unsqueeze(0))

    print(out.shape)
    print(out)


if __name__ == '__main__':
    main()