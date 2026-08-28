import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from statistics import mean

from get_data import PointMazeDataset
from define_model import PointMazeTranslationPredictor

def calc_ma(data: list, n: int):
    ma = [mean(data[i:i + n]) for i in range(len(data) - n + 1)]
    return ma


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
    N_data = 128000
    out_features = 4
    # --- training
    epochs = 20
    lr = 1e-4
    bs = 64
    lam = 0.001


    dataset = PointMazeDataset(N=N_data)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    predictor = PointMazeTranslationPredictor(out_features=out_features)
    optim = torch.optim.Adam(params=predictor.parameters(), lr=lr)

    losses = []
    losses_pred = []
    losses_sigreg = []
    fig, ax = plt.subplots()

    for epoch in range(epochs):
        for i, (curr_state, rand_action, next_state) in enumerate(dataloader):
            pred_out = predictor(curr_state, rand_action)
            loss_pred = ((next_state - pred_out) ** 2).mean()
            loss_sigreg = get_loss_sigreg(pred_out)
            loss = loss_pred + lam * loss_sigreg
            # loss = loss_pred
            # loss_sigreg = torch.tensor(0)

            loss.backward()
            optim.step()
            optim.zero_grad()

            losses.append(loss.item())
            losses_pred.append(loss_pred.item())
            losses_sigreg.append(lam * loss_sigreg.item())
            # if i % 10 == 0:
            #     ax.cla()
            #     plot_field_online(ax, encoder, side)
            #     plt.pause(0.01)
            print(f'\r[epoch {epoch}/{epochs}][batch {i}/{len(dataloader)}] loss = {loss.item()}', end='')

    # save the weights
    torch.save(predictor.state_dict(), 'point_maze_predictor.pt')
    print('--- saved ---')

    # with torch.no_grad():
    #     curr_state, rand_action, next_state = next(iter(dataloader))
    #     data = encoder(curr_state)
    #     pca_plot(data)
    losses_plot(losses, losses_pred, losses_sigreg)

def main():
    train_procedure()


if __name__ == '__main__':
    main()

