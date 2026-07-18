import torch
import torchvision
from torch.nn import functional as F
from my_dataset import train_dataloader
from matplotlib import pyplot as plt
from utils import corrupt
from create_model import BasicUNet


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BasicUNet()
    net.load_state_dict(torch.load("ddpm_simple/model.pt", map_location=device))
    net.to(device)
    net.eval()

    #@markdown Visualizing model predictions on noisy inputs:
    # Fetch some data
    x, y = next(iter(train_dataloader))
    x = x[:8] # Only using the first 8 for easy plotting

    # Corrupt with a range of amounts
    amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
    noised_x = corrupt(x, amount)

    # Get the model predictions
    with torch.no_grad():
      preds = net(noised_x.to(device)).detach().cpu()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title('Input data')
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
    axs[1].set_title('Corrupted data')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
    axs[2].set_title('Network Predictions')
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')
    plt.show()