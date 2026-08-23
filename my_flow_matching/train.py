import torch
import numpy as np
import modal
import wandb
import matplotlib.pyplot as plt

from define_model import MLP
from get_data import create_data
from config import config


image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install(
    "ffmpeg"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb", "torchcodec", "lpips"
).add_local_python_source(
    "define_model", "get_data", "config",
)
app = modal.App("mastering-pytorch")


def get_x_0(x_1, device):
    # x_0 = x_1.detach().clone().to(device)
    # x_0 = x_0 + torch.rand_like(x_0).to(device)
    # for _ in range(100):
    #     x_0 = x_0 + 0.01 * torch.rand_like(x_0).to(device)
    x_0 = torch.randn_like(x_1).to(device)
    return x_0


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='my_dit',
        config=config,
        mode='disabled'
    )

    data = torch.tensor(create_data()).to(device)

    model = MLP(layers=config.layers, channels=config.channels).to(device)
    params = [p.numel() for p in model.parameters()]
    print(f'num params: {sum(params)}')
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = []

    for training_step in range(config.training_steps):
        print(
            f'\r[{training_step / config.training_steps * 100: .2f}]{training_step}/{config.training_steps} '
            f'- loss: {float(np.mean(losses[-1000:])): .3f}',
            end=''
        )
        x_1  = data[torch.randint(data.shape[0], (config.bs,))]
        x_0 = get_x_0(x_1, device)
        target = x_1 - x_0
        t = torch.rand(x_1.shape[0]).to(device)
        xt = (1 - t[:, None]) * x_0 + t[:, None] * x_1
        pred = model(xt, t)
        loss = ((target - pred) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        losses.append(loss.item())
        # loss_p90 = ((target - pred) ** 2)[t > 0.9].mean().item()
        # if loss_p90 > 0:
        #     losses.append(loss_p90)
        # if training_step % 100 == 0:
        #     plt.cla()
        #     plt.plot(losses)
        #     plt.pause(0.001)

    # save
    # torch.save(model.state_dict(), 'state_dict.pt')
    # print('\n--- saved ---')
    yield {
        'type': 'final',
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    }

    # plt.show()


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def modal_train():
    for chunk in train():
        yield chunk


@app.local_entrypoint()
def modal_main():
    for chunk in modal_train.remote_gen():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict = chunk["state_dict"]
            torch.save(state_dict, f'state_dict.pt')
            print(f'--- saved ---')
    print(f'--- finished ---')


def main():
    for chunk in train():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict = chunk["state_dict"]
            torch.save(state_dict, f'state_dict.pt')
            print(f'--- saved ---')
    print(f'--- finished ---')


if __name__ == '__main__':
    main()
