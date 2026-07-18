import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
from create_model import ClassConditionedUnet
from get_data import dataset
import modal


image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "create_model", "get_data"
).add_local_file(
    "model.pt", remote_path="/root/model.pt"
)
app = modal.App("mastering-pytorch-ddpm")

num_train_timesteps = 1000


def train():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule='squaredcos_cap_v2'
    )
    net = ClassConditionedUnet().to(device)
    net.load_state_dict(torch.load("model.pt", map_location=device))
    # net.load_state_dict(torch.load("/root/model.pt", map_location=device))
    #@markdown Sampling some different digits:
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(80, 1, 28, 28).to(device)
    y = torch.tensor([[i]*8 for i in range(10)]).flatten().to(device)
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        # Get model pred
        with torch.no_grad():
            residual = net(x, t, y)  # Again, note that we pass in our labels y
        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
        yield x.detach().cpu().clip(-1, 1)


@app.function(image=image, gpu="A10G", timeout=36000, secrets=[modal.Secret.from_name("wandb-secret")])
def modal_train():
    for c in train():
        yield c


@app.local_entrypoint()
def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    images, counter = [], 0
    for chunk in modal_train.remote_gen():
        images.append(chunk)
        print(f'{counter=}')
        counter += 1
    print('here')

    def update(i):
        ax.cla()
        ax.imshow(torchvision.utils.make_grid(images[i], nrow=8)[0], cmap='Greys')
        ax.set_title(f'step {i}')
        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=len(images), interval=100)

    # GIF (no external deps needed beyond Pillow, which you likely already have)
    anim.save(f'training_{num_train_timesteps}.gif', writer=animation.PillowWriter(fps=10))

    # MP4 (needs ffmpeg installed and on PATH)
    anim.save(f'training_{num_train_timesteps}.mp4', writer=animation.FFMpegWriter(fps=10))

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    for chunk in train():
        ax.cla()
        ax.imshow(torchvision.utils.make_grid(chunk, nrow=8)[0], cmap='Greys')
        plt.pause(0.5)
    plt.show()