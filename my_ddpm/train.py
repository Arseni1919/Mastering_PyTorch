import torch
from torch.utils.data import DataLoader
from get_data import dataset
from config import config
from diffusers import DDPMScheduler
from define_model import MyUNetClassConditionedModel
from torch.optim import Adam
from torch.nn import MSELoss
import wandb
import modal
import utils


image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "define_model", "get_data", "config", "utils"
)
app = modal.App("mastering-pytorch-my_ddpm")



def train_epoch(device, dataloader, noise_scheduler, net, loss_fn, optimizer):
    log_losses = []
    n = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device) * 2 - 1
        y = y.to(device)
        noise = torch.randn_like(x).to(device)
        rand_timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, rand_timesteps)
        pred = net(noisy_x, rand_timesteps, y)
        loss = loss_fn(pred, noise)
        net.zero_grad()
        loss.backward()
        optimizer.step()

        log_losses.append(loss.item())
        wandb.log({"loss": loss.item()})
        print(f'\r[{i}/{n}] loss={loss.item():.3f}', end='')

    return log_losses


def train_process():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    noise_scheduler: DDPMScheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps, beta_schedule='squaredcos_cap_v2'
    )
    example_x, _ = dataset[0]
    net = MyUNetClassConditionedModel(example_x=example_x).to(device)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = MSELoss()

    wandb.init(project='my_ddpm', config=config)
    all_losses = []

    for epoch in range(config.n_epochs):
        log_losses = train_epoch(device, dataloader, noise_scheduler, net, loss_fn, optimizer)

        avg_loss = sum(log_losses) / len(log_losses)
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})
        all_losses.extend(log_losses)
        msg = f'\n--- {epoch=} | {avg_loss=} ---'
        print(msg)
        yield {'type': 'log', 'msg': msg}
    wandb.finish()
    yield {'type': 'final', 'state_dict': net.cpu().state_dict(), 'all_losses': all_losses}


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def modal_train():
    for out in train_process():
        yield out



@app.local_entrypoint()
def modal_main():
    for out in modal_train.remote_gen():
        if out['type'] == 'log':  # print(f'{out["msg"]}')
            continue
        state_dict, all_losses = out["state_dict"], out["all_losses"]
        torch.save(state_dict, 'saved_model.pt')
        utils.plot_losses(all_losses)
        print(f'--- finished ---')


def main():
    for chunk in train_process():
        if chunk['type'] == 'log':   # print(f'{out["msg"]}')
            continue
        state_dict, all_losses = chunk["state_dict"], chunk["all_losses"]
        torch.save(state_dict, 'saved_model.pt')
        utils.plot_losses(all_losses)
        print(f'--- finished ---')


if __name__ == '__main__':
    main()