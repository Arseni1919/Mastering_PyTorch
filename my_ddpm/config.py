from dataclasses import dataclass, field

@dataclass
class Config:
    num_classes: int = 10
    embedding_dim: int = 8

config = Config()
