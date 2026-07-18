import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from create_model import ClassConditionedUnet
from get_data import dataset
import wandb
import modal

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "create_model", "get_data"
)
app = modal.App("mastering-pytorch-ddpm_simple")


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def train():
    wandb.init(project="ddpm_simple-mnist", config={"n_epochs": 10, "batch_size": 128, "lr": 1e-3})
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    # Create a scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    #@markdown Training loop (10 Epochs):

    # Redefining the dataloader to set the batch size higher than the demo of 8
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # How many runs through the data should we do?
    n_epochs = 10

    # Our network
    net = ClassConditionedUnet().to(device)

    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y

            # Calculate the loss
            loss = loss_fn(pred, noise) # How close is the output to the noise

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())
            wandb.log({"loss": loss.item()})

        # Print out the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(losses[-100:])/100
        print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})

    wandb.finish()
    return net.cpu().state_dict(), losses

@app.local_entrypoint()
def main():
    state_dict, losses = train.remote()
    torch.save(state_dict, "model.pt")
    plt.plot(losses)
    # plt.ylim(0, 0.1)
    plt.show()


if __name__ == '__main__':
    state_dict, losses = train()
    torch.save(state_dict, "model.pt")
    plt.plot(losses)
    # plt.ylim(0, 0.1)
    plt.show()


