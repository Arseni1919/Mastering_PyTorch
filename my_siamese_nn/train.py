import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config import config
from get_data import dataset
from define_model import Classifier, HyperNet, Switcher
from torch.func import functional_call


def sample_pixels(main_net: nn.Module, hyper_net: nn.Module, gray_image: torch.Tensor, side_size: int):
    names = [name for name, _ in main_net.named_parameters()]
    numels = [p.numel() for _, p in main_net.named_parameters()]
    shapes = [p.shape for _, p in main_net.named_parameters()]
    weights = hyper_net(gray_image.flatten().unsqueeze(0))
    chunks = weights.split(numels, dim=-1)
    params = {}
    for name, chunk, shape in zip(names, chunks, shapes):
        params[name] = chunk.reshape(shape)
    coordinates = torch.cartesian_prod(torch.arange(side_size), torch.arange(side_size))
    out = functional_call(main_net, params, coordinates.float()).detach()
    return coordinates, out


def main():
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    grayscale_transform = torchvision.transforms.Grayscale()

    gray_image = grayscale_transform(dataset[0][0])
    ch, h_, w_ = gray_image.shape
    classifier = Classifier(in_dim=2, hidden_dim=config.classifier_hidden_dim, out_dim=3, num_layers=10)
    hyper_net = HyperNet(input_net=classifier, in_dim=ch * h_ * w_, hidden_dim=config.hyper_hidden_dim)
    switcher = Switcher(classifier, hyper_net)

    optimizer = torch.optim.Adam(hyper_net.parameters(), lr=config.lr)

    losses = []
    epoch_losses = []
    counter = 0
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    for epoch in range(config.epochs):
        epoch_losses_temp = []
        for batch_indx, (images, labels) in enumerate(dataloader):
            gray_images = grayscale_transform(images)
            coords = torch.randint(0, w_, (config.batch_size, 2))
            out = switcher(gray_images.flatten(1), coords.float())
            idx = torch.arange(images.shape[0])  # (16,)
            y, x = coords[:, 0], coords[:, 1]  # each (16,)
            target = images[idx, :, y, x]
            loss = F.mse_loss(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            counter += 1
            print(f'\r[{counter}][{batch_indx}/{len(dataloader)}] loss={loss.item()}', end='')
            if counter > 50:
                losses.append(loss.item())
                epoch_losses_temp.append(loss.item())
                ax[0].cla()
                ax[0].plot(losses)
            if counter % 100 == 0:
                rand_image, _ = dataset[random.randint(0, len(dataset) - 1)]
                ax[1].imshow(rand_image.permute(1, 2, 0).numpy())
                rand_gray_image = grayscale_transform(rand_image)
                ax[2].imshow(rand_gray_image.permute(1, 2, 0).numpy(), cmap='gray')
                sample_coords, sample_out = sample_pixels(classifier, hyper_net, rand_gray_image, side_size=h_)
                sample_out = sample_out.reshape(h_, w_, 3).numpy()
                # sample_out = sample_out.reshape(3, h_, w_).permute(1, 2, 0).numpy()
                ax[3].imshow(sample_out)
            plt.pause(0.01)
        epoch_losses.append(np.mean(epoch_losses_temp))

    plt.show()



if __name__ == '__main__':
    main()