import torch
from get_data import dataset
from define_lora import load_model
from torch.utils.data import DataLoader
from config import config
import wandb
import modal

image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install(
    "ffmpeg"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "tqdm", "wandb", "transformers",
).add_local_python_source(
    "define_lora", "get_data", "config"
).add_local_file(
    'tinyshakespeare.txt', remote_path='/root/tinyshakespeare.txt'
)
app = modal.App("mastering-pytorch")


def train_procedure():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='my_gpt_lora',
        config=config,
        # mode='disabled'
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model, tokenizer = load_model(
        config.model_name, config.target_modules, config.lora_rank, config.lora_alpha, config.lora_dropout, device
    )
    print(f'encoder num params: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate
    )
    n_datapoints = len(dataloader)

    losses = []
    for epoch in range(config.n_epochs):
        epoch_losses = []
        for data_idx, x in enumerate(dataloader):
            x: torch.Tensor = x.to(device)
            outputs = model(x, labels=x)
            loss = outputs.loss
            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({'loss': loss.item()})
            print(f'\r[{epoch}/{config.n_epochs} | {data_idx}/{n_datapoints}] loss={loss.item()}', end='')
            if data_idx % 500 == 0 and data_idx != 0:
                yield {
                    'type': 'final',
                    'state_dict_lora': {k: v.cpu() for k, v in model.state_dict().items() if 'A' in k or 'B' in k},
                }
            else:
                yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'\n[{epoch=}] avr loss={avr_epoch_loss: .3f}')
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {
        'type': 'final',
        'state_dict_lora': {k: v.cpu() for k, v in model.state_dict().items() if 'A' in k or 'B' in k},
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def modal_train():
    for chunk in train_procedure():
        yield chunk


@app.local_entrypoint()
def modal_main():
    for chunk in modal_train.remote_gen():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict_lora = chunk["state_dict_lora"]
            torch.save(state_dict_lora, f'state_dict_lora.pt')
    print(f'--- finished ---')


def main():
    for chunk in train_procedure():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict_lora = chunk["state_dict_lora"]
            torch.save(state_dict_lora, f'state_dict_lora.pt')
    print(f'--- finished ---')


if __name__ == '__main__':
    main()