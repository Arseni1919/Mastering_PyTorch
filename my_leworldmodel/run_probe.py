"""Linear probe: can a plain linear map read the agent's (row, col) out of the latent?

This is the gate for planning -- if the probe is weak, no planner can work, because
latent distance carries no positional information.
"""
import torch

from define_model import Encoder


def all_cells(side: int):
    states, coords = [], []
    for row in range(side):
        for col in range(side):
            state = torch.zeros(1, side, side)
            state[0, row, col] = 1.
            states.append(state)
            coords.append((row, col))
    return torch.stack(states), torch.tensor(coords, dtype=torch.float)


def linear_probe(encoder, side: int = 15):
    states, coords = all_cells(side)
    with torch.no_grad():
        z = encoder(states)

    design = torch.cat([z, torch.ones(len(z), 1)], dim=1)          # bias column
    weights = torch.linalg.lstsq(design, coords).solution
    residual = ((design @ weights - coords) ** 2).sum(0)
    r2 = 1 - residual / ((coords - coords.mean(0)) ** 2).sum(0)

    # a probe only PROJECTS position out; L2 distance sums over every dim, so any
    # off-position variance is noise the planner cannot ignore. Regress the other way
    # -- latent ON position -- to see how much of the latent position actually accounts for.
    pos_design = torch.cat([coords, torch.ones(len(coords), 1)], dim=1)
    fit = pos_design @ torch.linalg.lstsq(pos_design, z).solution
    explained = (fit - fit.mean(0)).pow(2).sum() / (z - z.mean(0)).pow(2).sum()

    latent_d = torch.cdist(z, z)
    grid_d = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1).float()
    off_diag = ~torch.eye(len(coords), dtype=torch.bool)
    dcorr = torch.corrcoef(torch.stack([latent_d[off_diag], grid_d[off_diag]]))[0, 1]

    return r2[0].item(), r2[1].item(), explained.item(), dcorr.item()


def main():
    side = 15
    out_features = 2

    encoder = Encoder(in_features=side**2, out_features=out_features)
    encoder.load_state_dict(torch.load('encoder.pt'))
    encoder.eval()

    r2_row, r2_col, explained, dcorr = linear_probe(encoder, side)
    print(f'linear probe R2:  row {r2_row:.3f} | col {r2_col:.3f}')
    print(f'latent variance explained by position: {explained:.1%}')
    print(f'corr(latent distance, grid distance):  {dcorr:.3f}')
    print()
    print('gate for a distance-based planner, BOTH are needed:')
    print('  R2 > 0.9      -- position is linearly readable at all')
    print('  dcorr > 0.9   -- and it dominates the latent, so L2 distance tracks it')
    print('  (d=16 scored R2 0.99 with dcorr 0.82 and planned at 57%: readable, but drowned)')


if __name__ == '__main__':
    main()
