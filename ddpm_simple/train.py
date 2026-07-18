import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import corrupt, app, image
from my_dataset import dataset
from create_model import BasicUNet


@app.function(image=image, gpu="A10G", timeout=3600)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_epochs = 10
    net = BasicUNet()
    net.to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for epoch in range(n_epochs):
        for x, y in train_dataloader:
            x = x.to(device) # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount) # Create our noisy x
            pred = net(noisy_x)
            loss = loss_fn(pred, x) # How close is the output to the true 'clean' x?
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
        print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    return net.cpu().state_dict(), losses


@app.local_entrypoint()
def main():
    state_dict, losses = train.remote()
    torch.save(state_dict, "ddpm_simple/model.pt")
    plt.plot(losses)
    plt.ylim(0, 0.1)
    plt.show()
