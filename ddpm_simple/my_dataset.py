import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import corrupt

dataset = torchvision.datasets.MNIST(
    root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    x, y = next(iter(train_dataloader))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    plt.show()

    # Plotting the input data
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].set_title('Input data')
    axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    # Adding noise
    amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
    noised_x = corrupt(x, amount)
    # Plotting the noised version
    axs[1].set_title('Corrupted data (-- amount increases -->)')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')
    plt.show()
