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
        if len(x.shape) == 4:
            bs, ch, h_, w_ = x.shape
            x = x.reshape((bs, -1))
        x = self.mlp(x)
        return x


class PointMazeTranslationPredictor(nn.Module):
    def __init__(self, out_features: int = 4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        return x + self.net(action)


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





class ConvEncoder(nn.Module):
    """Stacked RGB frames -> latent vector.

    Input is (B, 3*history, 64, 64): `history` consecutive frames concatenated on the
    channel axis. The first conv sees all of them at once, so it can form temporal
    differences between planes -- that is where velocity comes from, and a single frame
    cannot supply it (predicting the next position from one frame gives R2 0.07, versus
    0.97 once velocity is available).

    The head ends in BatchNorm, NOT LayerNorm: SIGReg shapes the distribution ACROSS the
    batch per feature, and LayerNorm (which normalises within a sample) would mask exactly
    what SIGReg is trying to create.
    """

    def __init__(self, in_channels: int = 9, latent_dim: int = 8, width: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),      # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),        # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),    # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=2, padding=1),    # 8  -> 4
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * 4, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.head(self.conv(x))


class LatentPredictor(nn.Module):
    """z_{t+1} = z_t + mlp([z_t, a]) -- a residual latent dynamics model.

    Conditioning on z_t as well as the action is not optional here. PointMaze is second
    order: the action is a force, and where the ball goes is dominated by the velocity it
    already carries. Predicting the next position from action alone gives R2 0.07, versus
    0.97 once the current state is available.

    The final layer is zero-initialised so the model starts as the exact identity
    (z_{t+1} = z_t) and learns the correction from there -- a near-correct starting point
    for a small timestep, and it keeps early training stable.
    """

    def __init__(self, latent_dim: int = 8, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        return z + self.mlp(torch.cat([z, action], dim=-1))


class LinearPredictor(nn.Module):
    """z_{t+1} = z_t + W [z_t, a]  -- linear latent dynamics (A = I + W_z, B = W_a).

    The minimal predictor class that can still express second-order dynamics: position
    changes by velocity, which a constant per-action offset cannot represent but a linear
    map can (linear regression from state+action predicts the next state at R2 ~0.98 here).

    The point of keeping it linear is the constraint it puts on the ENCODER. If the
    predictor can only apply a linear map, the prediction loss is satisfiable only when the
    encoder is an affine function of the true state -- and affine means the latent is
    straight, so Euclidean distance between latents tracks real distance. That is what made
    the grid toy's translation predictor work, and what a flexible MLP gives up.

    Its limit: wall collisions are genuinely non-linear and this cannot express them.
    """

    def __init__(self, latent_dim: int = 8, action_dim: int = 2):
        super().__init__()
        self.lin = nn.Linear(latent_dim + action_dim, latent_dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        return z + self.lin(torch.cat([z, action], dim=-1))


class PositionHead(nn.Module):
    """latent -> (x, y): a read-out head used as the PLANNING COST.

    Planning by raw latent distance fails here for a concrete reason: the goal observation
    is the ball at REST, so z_goal carries zero velocity, and the latent encodes velocity
    too. Minimising ||z - z_goal|| therefore asks the planner to arrive AND stop, penalising
    the very motion that makes progress. Measured: 40% success with the raw latent cost,
    85-90% once the cost compares position only.

    Trained on the ground-truth states the dataset already stores, so this is a SUPERVISED
    component -- the rest of the world model is not.
    """

    def __init__(self, latent_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor):
        return self.mlp(z)
