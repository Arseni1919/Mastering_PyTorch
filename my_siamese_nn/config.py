from dataclasses import dataclass


@dataclass
class Config:
    num_layers: int = 15
    classifier_hidden_dim: int = 128
    hyper_hidden_dim: int = 128
    batch_size: int = 16
    epochs: int = 10
    p_train: int = 100
    lr: float = 0.001


config = Config()
