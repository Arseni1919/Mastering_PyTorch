from dataclasses import dataclass


@dataclass
class Config:
    num_layers: int = 10


config = Config()
