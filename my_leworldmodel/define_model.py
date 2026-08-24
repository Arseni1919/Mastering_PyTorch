import torch
import torch.nn as nn

from get_data import get_nav_data


class Encoder(nn.Module):
    def __init__(self, c_in: int = 1, side: int = 10, hidden: int = 512, in_features:int = 100, out_features: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x: torch.Tensor):
        bs, ch, h_, w_ = x.shape
        x = x.reshape((bs, -1))
        x = self.mlp(x)
        return x


class Predictor(nn.Module):
    def __init__(self, out_features: int = 10, hidden: int = 256):
        super().__init__()
        self.action_emb = nn.Embedding(5, out_features)
        self.mlp = nn.Sequential(
            nn.Linear(out_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
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