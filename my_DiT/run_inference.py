from typing import Any
import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from define_model import DiTFullModel
import modal
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


def plot_image(image, timestamp, model_config, pause_time=0.01):
    plt.cla()
    plt.imshow(image)
    plt.title(
        f'timestamp: {timestamp + 1}\n'
        f'hidden_size={model_config.hidden_size} | '
        f'num_layers={model_config.num_layers} | '
        f'n_epochs={model_config.n_epochs}'
    )
    plt.pause(pause_time)


def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_size = 1 / config.num_inference_steps
    example_x, _ = dataset[0]
    bs = 10
    ch, height, width = example_x.shape
    net: DiTFullModel = DiTFullModel(
        side=config.side_size,
        path_side=config.patch_size,
        head_dim=config.hidden_size // config.num_heads,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        class_emb_size=config.class_emb_size,
        time_emb_size=config.time_emb_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        channels=config.channels
    )
    net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    net.eval()
    print(f'num params: {sum(p.numel() for p in net.parameters())}')
    x = torch.randn((bs, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    t = 0.0

    for _ in tqdm(range(config.num_inference_steps)):
        t_tensor = torch.ones((bs,)) * t
        with torch.no_grad():
            pred = net(x, t_tensor, y)
        x += step_size * pred
        t += step_size
        yield {'type': 'img', 'x': x.cpu(), 'timestamp': t}


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def modal_run_inference():
    for chunk in run_inference():
        yield chunk


@app.local_entrypoint()
def modal_main():
    x_list = []
    for chunk in modal_run_inference.remote_gen():
        if chunk['type'] == 'img':
            x: torch.Tensor | Any = chunk['x']
            show_x = x.permute(0, 2, 3, 1).reshape(10*config.side_size, config.side_size, 1).numpy()
            show_x = (show_x + 1) / 2
            x_list.append(show_x)
    print('Start to show...')
    for t, show_x in enumerate(x_list):
        plot_image(show_x, t, config, pause_time=0.05)
    plt.show()


def main():
    x_list = []
    for chunk in run_inference():
        if chunk['type'] == 'img':
            x: torch.Tensor | Any = chunk['x']
            show_x = x.cpu().detach().permute(0, 2, 3, 1).reshape(10*config.side_size, config.side_size, 1).numpy()
            show_x = (show_x + 1) / 2
            x_list.append(show_x)
    print('Start to show...')
    for t, show_x in enumerate(x_list):
        plot_image(show_x, t, config, pause_time=0.05)
    plt.show()


if __name__ == '__main__':
    main()