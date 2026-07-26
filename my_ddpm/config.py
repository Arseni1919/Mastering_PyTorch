from dataclasses import dataclass, field

@dataclass
class Config:
    # define model
    num_classes: int = 10
    class_emb_size: int = 4
    time_emb_size: int = 32
    num_groups: int = 4
    base_channels: int = 8
    has_attn: bool = False
    # train
    n_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_train_timesteps: int = 1000
    # daploy
    onnx_path: str = 'model.onnx'

config = Config()
