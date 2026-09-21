"""JEPA world model on RGB observations of PointMaze.

Unlike train_point_maze.py (raw state vectors, no encoder), here a ConvEncoder is learned
jointly with the predictor, so the latent CAN collapse -- which is what SIGReg is for.

Three things this file does differently from the state-vector version, each for a reason
that cost us a debugging session:
  1. SIGReg is applied to BOTH branches (encoder output and target), not just the
     prediction. Without a stop-gradient the target is a live gradient path into the
     encoder; leaving it unregularised leaves the one branch that can collapse unwatched.
  2. The target is NOT detached. That is LeWorldModel's defining choice -- SIGReg alone
     prevents collapse.
  3. z.std is logged every step. Collapse looks like loss_pred falling while z.std decays,
     and the loss curve alone will not show it.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from statistics import mean

from get_data import PointMazeRGBDataset, get_point_maze_rgb_env
from define_model import ConvEncoder, LatentPredictor, LinearPredictor, PositionHead


def calc_ma(data: list, n: int):
    return [mean(data[i:i + n]) for i in range(len(data) - n + 1)]


def losses_plot(losses_pred, losses_sigreg, z_stds, n: int = 100):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, data, title in [(ax[0], losses_pred, 'loss_pred'),
                           (ax[1], losses_sigreg, 'sigreg (1.0 == Gaussian)'),
                           (ax[2], z_stds, 'z.std  (-> 0 means collapse)')]:
        x = list(range(len(data)))
        a.plot(x, data, alpha=0.4)
        if len(data) >= n:
            a.plot(x[n - 1:], calc_ma(data, n), color='red')
        a.set_title(title)
    ax[0].set_yscale('log')
    plt.tight_layout()
    plt.show()


def get_loss_sigreg(vectors: torch.Tensor, num_slices: int = 256,
                    n_points: int = 17, t_max: float = 5.0):
    """Epps-Pulley normality test on random 1-D projections of the latents.

    Reads ~1.0 when the latents really are Gaussian, at any batch size or latent dim --
    the P scaling makes the statistic's null distribution size-independent. So the value
    is absolute: 1 means Gaussian, 10 means clearly not.
    """
    P, D = vectors.shape

    directions = torch.randn(D, num_slices, device=vectors.device, dtype=vectors.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True)
    projections = vectors @ directions

    t = torch.linspace(-t_max, t_max, n_points, device=vectors.device, dtype=vectors.dtype)
    target_cf = torch.exp(-0.5 * t ** 2)

    phase = projections[..., None] * t
    empirical_cf = torch.exp(1j * phase).mean(0)

    diff = empirical_cf - target_cf
    weighted_err = (diff.real ** 2 + diff.imag ** 2) * target_cf
    per_slice = P * torch.trapezoid(weighted_err, t)

    return per_slice.mean()


def make_probe_grid(env, transform, history: int = 3, per_cell: int = 5):
    """A grid of maze positions, each rendered as a stationary observation.

    The continuous analogue of enumerating every cell in the toy grid world: we sample
    points across the free space, teleport the ball to each with ZERO velocity, render,
    and build the same stacked observation the encoder sees during training (the frame
    repeated `history` times -- a ball at rest).
    """
    import mujoco

    pe = env.unwrapped.point_env
    maze = env.unwrapped.maze
    obs_list, xy_list = [], []

    for row in range(len(maze.maze_map)):
        for col in range(len(maze.maze_map[0])):
            if maze.maze_map[row][col] != 0:          # 1 == wall
                continue
            cx, cy = maze.cell_rowcol_to_xy(np.array([row, col]))
            s = maze.maze_size_scaling
            for dx in np.linspace(-0.35, 0.35, per_cell) * s:
                for dy in np.linspace(-0.35, 0.35, per_cell) * s:
                    pe.data.qpos[:2] = [cx + dx, cy + dy]
                    pe.data.qvel[:2] = 0.0
                    mujoco.mj_forward(pe.model, pe.data)
                    frame = transform(env.render().copy())
                    obs_list.append(torch.cat([frame] * history, dim=0))
                    xy_list.append([cx + dx, cy + dy])

    return torch.stack(obs_list), np.array(xy_list)


def plot_field_online(ax, encoder, probe_obs, probe_xy):
    """The maze as the encoder sees it.

    The latent is >2-D and its axes are meaningless (SIGReg makes it isotropic, and an
    isotropic Gaussian is rotation invariant), so we read the position plane out of it:
    least-squares fit latent -> (x, y) on the probe grid, then scatter each point at its
    decoded position. A working encoder reproduces the U-maze; a collapsing one gives mush.
    """
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        z = encoder(probe_obs)
    if was_training:
        encoder.train()

    xy = torch.tensor(probe_xy, dtype=torch.float32)
    design = torch.cat([z, torch.ones(len(z), 1)], dim=1)
    decoded = (design @ torch.linalg.lstsq(design, xy).solution).numpy()

    for a, c, name in [(ax[0], probe_xy[:, 0], 'x'), (ax[1], probe_xy[:, 1], 'y')]:
        a.cla()
        sc = a.scatter(decoded[:, 0], decoded[:, 1], c=c, cmap='viridis', s=16)
        a.set_title(f'decoded position, coloured by true {name}')
        a.set_xlabel('decoded x'); a.set_ylabel('decoded y')
        a.set_aspect('equal')
        # one colourbar per axis, created once and reused so repeated calls do not stack them
        if not hasattr(a, '_cbar'):
            a._cbar = a.figure.colorbar(sc, ax=a)
        else:
            a._cbar.update_normal(sc)
        a._cbar.set_label(f'true {name}')


def train_procedure(visualize: bool = True, N_data: int = 5000):
    latent_dim = 8
    history = 3
    frameskip = 5
    predictor_type = 'mlp'   # 'linear' straightens the latent; 'mlp' fits the dynamics better
    # --- training
    epochs = 20
    lr = 3e-4
    bs = 128
    lam = 0.01

    dataset = PointMazeRGBDataset(N=N_data, frameskip=frameskip, history=history)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    encoder = ConvEncoder(in_channels=3 * history, latent_dim=latent_dim)
    if predictor_type == 'linear':
        predictor = LinearPredictor(latent_dim=latent_dim, action_dim=2)
    else:
        predictor = LatentPredictor(latent_dim=latent_dim, action_dim=2)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optim = torch.optim.Adam(params=params, lr=lr)

    losses_pred, losses_sigreg, z_stds = [], [], []

    probe_env = fig = ax = probe_obs = probe_xy = None
    if visualize:
        probe_env = get_point_maze_rgb_env()
        probe_env.reset(seed=0)
        probe_obs, probe_xy = make_probe_grid(probe_env, dataset.transform, history=history)
        print(f'probe grid: {len(probe_obs)} positions')
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        plt.ion()

    for epoch in range(epochs):
        for i, (curr_obs, action, next_obs) in enumerate(dataloader):
            enc_out = encoder(curr_obs)
            pred_out = predictor(enc_out, action)
            target = encoder(next_obs)          # NOT detached -- SIGReg is the only anti-collapse

            loss_pred = ((target - pred_out) ** 2).mean()
            loss_sigreg = get_loss_sigreg(torch.cat([enc_out, target], dim=0))
            loss = loss_pred + lam * loss_sigreg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad()

            z_std = enc_out.std(0).mean().item()
            losses_pred.append(loss_pred.item())
            losses_sigreg.append(loss_sigreg.item())
            z_stds.append(z_std)
            if visualize and i % 50 == 0:
                plot_field_online(ax, encoder, probe_obs, probe_xy)
                fig.suptitle(f'epoch {epoch}, batch {i}')
                plt.pause(0.01)
            print(f'\r[epoch {epoch}/{epochs}][batch {i}/{len(dataloader)}] '
                  f'pred {loss_pred.item():.5f}  sigreg {loss_sigreg.item():6.2f}  '
                  f'lam*sigreg {lam * loss_sigreg.item():.5f}  z.std {z_std:.3f}', end='')
        print()

    if visualize:
        plt.ioff()
        probe_env.close()
    # --- position head: the planning cost (see PositionHead for why raw latent distance fails)
    encoder.eval()
    with torch.no_grad():
        z_all = torch.cat([encoder(b[0]) for b in DataLoader(dataset, batch_size=256)])
    xy_all = torch.stack([dataset.get_states(i)[0] for i in range(len(dataset))])[:, :2]
    head = PositionHead(latent_dim=latent_dim)
    head_optim = torch.optim.Adam(head.parameters(), lr=1e-3)
    split = int(0.8 * len(z_all))
    for _ in range(4000):
        i = torch.randint(0, split, (256,))
        ((head(z_all[i]) - xy_all[i]) ** 2).mean().backward()
        head_optim.step(); head_optim.zero_grad()
    with torch.no_grad():
        r2 = 1 - ((head(z_all[split:]) - xy_all[split:]) ** 2).sum(0) / \
                 ((xy_all[split:] - xy_all[split:].mean(0)) ** 2).sum(0)
    print(f'position head held-out R2: {r2.numpy().round(3)}')
    torch.save(head.state_dict(), f'point_maze_rgb_position_head_{predictor_type}.pt')
    encoder.train()

    torch.save(encoder.state_dict(), f'point_maze_rgb_conv_encoder_{predictor_type}.pt')
    torch.save(predictor.state_dict(), f'point_maze_rgb_{predictor_type}_predictor.pt')
    print('--- saved ---')

    # if visualize:
    losses_plot(losses_pred, losses_sigreg, z_stds)


def main():
    # visualize=True opens the live latent-field window; False is faster and headless.
    train_procedure(visualize=True, N_data=20000)


if __name__ == '__main__':
    main()
