import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from statistics import mean

from get_data import NavDataset
from define_model import Encoder, TranslationPredictor, Predictor

def calc_ma(data: list, n: int):
    ma = [mean(data[i:i + n]) for i in range(len(data) - n + 1)]
    return ma


def plot_field_online(ax, encoder, side):
    all_images = torch.eye(side * side).reshape(side * side, 1, side, side)
    encodings = encoder(all_images).detach().numpy()
    x = [i[0] for i in encodings]
    y = [i[1] for i in encodings]
    ax.scatter(x, y)


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


def losses_plot(losses, losses_pred, losses_sigreg, n: int = 100):
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

    ax[2].plot(x_list, losses_sigreg)
    ma_losses = calc_ma(losses_sigreg, n)
    ax[2].plot(ma_x_list, ma_losses, color='red')
    ax[2].set_title('lam * losses_sigreg')

    plt.show()


def get_loss_sigreg(vectors: torch.Tensor, num_slices: int = 256,
                    n_points: int = 17, t_max: float = 5.0):
    """Epps-Pulley normality test on random 1-D projections of the latents."""
    P, D = vectors.shape

    # fresh random unit directions every call: cheap coverage that compounds over training
    directions = torch.randn(D, num_slices, device=vectors.device, dtype=vectors.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True)
    projections = vectors @ directions                          # (P, num_slices)

    t = torch.linspace(-t_max, t_max, n_points, device=vectors.device, dtype=vectors.dtype)
    target_cf = torch.exp(-0.5 * t ** 2)                        # char. function of N(0, 1)

    phase = projections[..., None] * t                          # (P, num_slices, n_points)
    empirical_cf = torch.exp(1j * phase).mean(0)                # (num_slices, n_points)

    # target_cf doubles as the weight, which is what makes the integral converge
    diff = empirical_cf - target_cf
    weighted_err = (diff.real ** 2 + diff.imag ** 2) * target_cf
    per_slice = P * torch.trapezoid(weighted_err, t)            # (num_slices,)

    return per_slice.mean()


def train_procedure():
    side = 15
    N_data = 100000
    out_features = 2     # 2-D latent: the grid has only 2 degrees of freedom
    # --- training
    epochs = 20
    lr = 1e-4
    bs = 64
    lam = 0.001   # sigreg reads ~1.0 when the latent IS Gaussian, ~10-50 while it is not,
                 # and loss_pred is in [0, 1] (unit-variance latents), so lam*sigreg is
                 # comparable to loss_pred for lam in 0.01-0.1. Picked 0.01 by probe R2.


    dataset = NavDataset(N=N_data, side=side)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    encoder = Encoder(in_features=side**2, out_features=out_features)
    predictor = TranslationPredictor(out_features=out_features)
    # predictor = Predictor(out_features=out_features)
    params = [p for p in encoder.parameters()] + [p for p in predictor.parameters()]
    optim = torch.optim.Adam(params=params, lr=lr)

    losses = []
    losses_pred = []
    losses_sigreg = []
    fig, ax = plt.subplots()

    for epoch in range(epochs):
        for i, (curr_state, rand_action, next_state) in enumerate(dataloader):
            enc_out = encoder(curr_state)
            pred_out = predictor(enc_out, rand_action)
            target = encoder(next_state)          # NOT detached: SIGReg is the only anti-collapse
            loss_pred = ((target - pred_out) ** 2).mean()
            loss_sigreg = get_loss_sigreg(torch.cat([enc_out, target], dim=0))
            loss = loss_pred + lam * loss_sigreg
            # loss = loss_pred
            # loss_sigreg = 0

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch > 2:
                losses.append(loss.item())
                losses_pred.append(loss_pred.item())
                losses_sigreg.append(lam * loss_sigreg.item())
            if i % 10 == 0:
                ax.cla()
                plot_field_online(ax, encoder, side)
                plt.pause(0.01)
            print(f'\r[epoch {epoch}/{epochs}][batch {i}/{len(dataloader)}] loss = {loss.item()}', end='')

    # save the weights
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(predictor.state_dict(), 'predictor.pt')
    print('--- saved ---')

    plt.show()

    with torch.no_grad():
        curr_state, rand_action, next_state = next(iter(dataloader))
        data = encoder(curr_state)
        pca_plot(data)
        losses_plot(losses, losses_pred, losses_sigreg)

def main():
    train_procedure()


if __name__ == '__main__':
    main()

