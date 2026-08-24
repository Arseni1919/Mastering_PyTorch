import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from get_data import NavDataset
from define_model import Encoder, Predictor


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
    side = 10
    out_features = 10
    # --- training
    epochs = 20
    lr = 1e-4
    bs = 64
    lam = 1


    dataset = NavDataset(side=side)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    encoder = Encoder(out_features=out_features)
    predictor = Predictor(out_features=out_features)
    params = [p for p in encoder.parameters()] + [p for p in predictor.parameters()]
    optim = torch.optim.Adam(params=params, lr=lr)

    losses = []

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
            print(f'\r[epoch {epoch}/{epochs}][batch {i}/{len(dataloader)}] loss = {loss.item()}', end='')

    with torch.no_grad():
        curr_state, rand_action, next_state = next(iter(dataloader))
        data = encoder(curr_state)
        pca_plot(data)
        plt.plot(losses)
        plt.show()

def main():
    train_procedure()


if __name__ == '__main__':
    main()

