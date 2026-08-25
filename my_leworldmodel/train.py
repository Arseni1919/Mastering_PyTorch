import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from statistics import mean

from get_data import NavDataset
from define_model import Encoder, Predictor

def calc_ma(data: list, n: int):
    ma = [mean(data[i:i + n]) for i in range(len(data) - n + 1)]
    return ma


def pca_plot(data: torch.Tensor):

    data = data.numpy()
    data = data - data.mean(0)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(data)

    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.gca().set_aspect('equal')
    plt.show()


def losses_plot(losses, losses_pred, losses_segreg, n: int = 100):
    fig, ax = plt.subplots(1, 3)

    x_list = list(range(len(losses)))
    ma_x_list = x_list[n-1:]

    ax[0].plot(x_list, losses)
    ma_losses = calc_ma(losses, n)
    ax[0].plot(ma_x_list, ma_losses, color='red')
    ax[0].set_title('losses')

    ax[1].plot(x_list, losses_pred)
    ma_losses = calc_ma(losses_pred, n)
    ax[1].plot(ma_x_list, ma_losses, color='red')
    ax[1].set_title('losses_pred')

    ax[2].plot(x_list, losses_segreg)
    ma_losses = calc_ma(losses_segreg, n)
    ax[2].plot(ma_x_list, ma_losses, color='red')
    ax[2].set_title('losses_segreg')

    plt.show()

def get_loss_segreg(vectors: torch.Tensor, gamma=1.0, eps=1e-4,
                    var_coef=1.0, cov_coef=0.04):
    B, D = vectors.shape
    z = vectors - vectors.mean(0)

    # variance: push each dim's std across the batch up to gamma
    std = torch.sqrt(z.var(0) + eps)
    var_loss = torch.relu(gamma - std).mean()

    # covariance: decorrelate dimensions so they don't duplicate each other
    cov = (z.T @ z) / (B - 1)
    cov_loss = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / D

    return var_coef * var_loss + cov_coef * cov_loss


def train_procedure():
    side = 15
    out_features = 16
    # --- training
    epochs = 10
    lr = 1e-4
    bs = 64
    lam = 1


    dataset = NavDataset(side=side)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    encoder = Encoder(in_features=side**2, out_features=out_features)
    predictor = Predictor(out_features=out_features)
    params = [p for p in encoder.parameters()] + [p for p in predictor.parameters()]
    optim = torch.optim.Adam(params=params, lr=lr)

    losses = []
    losses_pred = []
    losses_segreg = []

    for epoch in range(epochs):
        for i, (curr_state, rand_action, next_state) in enumerate(dataloader):
            enc_out = encoder(curr_state)
            pred_out = predictor(enc_out, rand_action)
            with torch.no_grad():
                target = encoder(next_state)
            loss_pred = ((target - pred_out) ** 2).mean()
            loss_segreg = get_loss_segreg(enc_out)
            loss = loss_pred + lam * loss_segreg  # lam ~ 1.0 to start

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch > 2:
                losses.append(loss.item())
                losses_pred.append(loss_pred.item())
                losses_segreg.append(loss_segreg.item())
            print(f'\r[epoch {epoch}/{epochs}][batch {i}/{len(dataloader)}] loss = {loss.item()}', end='')

    # save the weights
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(predictor.state_dict(), 'predictor.pt')

    with torch.no_grad():
        curr_state, rand_action, next_state = next(iter(dataloader))
        data = encoder(curr_state)
        pca_plot(data)
        losses_plot(losses, losses_pred, losses_segreg)

def main():
    train_procedure()


if __name__ == '__main__':
    main()

