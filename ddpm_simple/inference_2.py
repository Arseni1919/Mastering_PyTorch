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
    net.load_state_dict(torch.load("model.pt", map_location=device))
    net.to(device)
    net.eval()
    #@markdown Sampling strategy: Break the process into 5 steps and move 1/5'th of the way there each time:
    n_steps = 5
    x = torch.rand(8, 1, 28, 28).to(device) # Start from random
    step_history = [x.detach().cpu()]
    pred_output_history = []

    for i in range(n_steps):
        with torch.no_grad(): # No need to track gradients during inference
            pred = net(x) # Predict the denoised x0
        pred_output_history.append(pred.detach().cpu()) # Store model output for plotting
        mix_factor = 1/(n_steps - i) # How much we move towards the prediction
        x = x*(1-mix_factor) + pred*mix_factor # Move part of the way there
        step_history.append(x.detach().cpu()) # Store step for plotting

    fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    axs[0,0].set_title('x (model input)')
    axs[0,1].set_title('model prediction')
    for i in range(n_steps):
        axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
        axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')
    plt.show()

    # @markdown Showing more results, using 40 sampling steps
    n_steps = 40
    x = torch.rand(64, 1, 28, 28).to(device)
    for i in range(n_steps):
        noise_amount = torch.ones((x.shape[0],)).to(device) * (1 - (i / n_steps))  # Starting high going low
        with torch.no_grad():
            pred = net(x)
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(torchvision.utils.make_grid(x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
    plt.show()