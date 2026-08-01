from dataclasses import dataclass

@dataclass
class Config:
    channels: int = 3
    side_size: int = 32
    base_channels: int = 128
    latent_dim: int = 256
    n_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3e-4
    kl_weight: float = 0.3
    num_interpolation_steps: int = 10
    num_stages: int = 2


config = Config()