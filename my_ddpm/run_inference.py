from typing import Any
import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from diffusers import DDPMScheduler
from define_model import MyUNetClassConditionedModel
import modal
import utils
from tqdm import tqdm


image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "define_model", "get_data", "config", "utils"
).add_local_file(
    "saved_model.pt", remote_path="/root/saved_model.pt"
)
app = modal.App("mastering-pytorch-my_ddpm")




def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    noise_scheduler: DDPMScheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps, beta_schedule='squaredcos_cap_v2'
    )
    example_x, _ = dataset[0]
    ch, height, width = example_x.shape
    net = MyUNetClassConditionedModel(example_x=example_x).to(device)
    net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    net.eval()
    x = torch.randn((10, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    for timestamp in tqdm(noise_scheduler.timesteps):
        adapted_timestamp = timestamp.expand(x.shape[0]).to(device)
        with torch.no_grad():
            pred = net(x, adapted_timestamp, y)
        x = noise_scheduler.step(pred, timestamp, x).prev_sample
        yield {'type': 'img', 'x': x, 'timestamp': timestamp.item()}


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def modal_run_inference():
    for chunk in run_inference():
        yield chunk


@app.local_entrypoint()
def modal_main():
    for chunk in modal_run_inference.remote_gen():
        if chunk['type'] == 'img':
            timestamp = chunk['timestamp']
            x: torch.Tensor | Any = chunk['x']
            if timestamp % 50 == 0:
                show_x = x.cpu().detach().permute(0, 2, 3, 1).reshape(280, 28, 1).numpy()
                show_x = (show_x + 1) / 2
                utils.plot_image(show_x)
    plt.show()


def main():
    for chunk in run_inference():
        if chunk['type'] == 'img':
            timestamp = chunk['timestamp']
            x: torch.Tensor | Any = chunk['x']
            if timestamp % 100 == 0:
                show_x = x.cpu().detach().permute(0, 2, 3, 1).reshape(280, 28, 1).numpy()
                show_x = (show_x + 1) / 2
                utils.plot_image(show_x)
    plt.show()


if __name__ == '__main__':
    main()
