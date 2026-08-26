import torch
import torch.nn as nn

from get_data import get_nav_data


class Encoder(nn.Module):
    def __init__(self, c_in: int = 1, side: int = 10, hidden: int = 512, in_features:int = 100, out_features: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x: torch.Tensor):
        bs, ch, h_, w_ = x.shape
        x = x.reshape((bs, -1))
        x = self.mlp(x)
        return x


class TranslationPredictor(nn.Module):
    """z_{t+1} = z_t + emb(a).

    An action can only TRANSLATE the latent, never reshape it. That constraint is what
    forces the encoder to lay the grid out as a lattice: the only way to predict every
    transition with a single per-action offset is for the latent to be an affine function
    of the agent's (row, col). The flexible MLP Predictor above has no such pressure --
    it can memorise an arbitrary scramble of the 225 states, which leaves latent distance
    meaningless for planning.
    """
    def __init__(self, out_features: int = 2):
        super().__init__()
        self.action_emb = nn.Embedding(5, out_features)
        nn.init.zeros_(self.action_emb.weight)

    def forward(self, x, action):
        return x + self.action_emb(action)


class Predictor(nn.Module):
    def __init__(self, out_features: int = 10, hidden: int = 16):
        super().__init__()
        self.action_emb = nn.Embedding(5, out_features)
        nn.init.zeros_(self.action_emb.weight)
        self.mlp = nn.Sequential(
            nn.Linear(out_features, hidden),
            nn.GELU(),
            # nn.Linear(hidden, hidden),
            # nn.GELU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x, action):
        h = x + self.action_emb(action)
        x = x + self.mlp(h)
        return x


def main():
    side = 10
    out_features = 10
    data = get_nav_data(side=side)
    encoder = Encoder(out_features=out_features)
    predictor = Predictor(out_features=out_features)

    for i, (curr_state, rand_action, next_state) in enumerate(data):
        enc_out = encoder(curr_state.reshape(1, 1, *curr_state.shape))
        print(enc_out.shape)
        pred_out = predictor(enc_out, rand_action)
        print(pred_out.shape)
        break


if __name__ == '__main__':
    main()



