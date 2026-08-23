from dataclasses import dataclass


@dataclass
class Config:
    lr = 1e-4
    training_steps = 10000
    bs = 64
    layers = 5
    channels = 512
    steps = 100


config = Config()