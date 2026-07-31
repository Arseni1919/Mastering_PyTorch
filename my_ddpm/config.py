from dataclasses import dataclass, field

@dataclass
class Config:
    # data
    side_size: int = 32
    # define model
    num_classes: int = 10
    class_emb_size: int = 4
    time_emb_size: int = 32
    num_groups: int = 4
    base_channels: int = 64
    has_attn: bool = False
    num_layers: int = 2
    # train
    n_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 5e-4
    num_train_timesteps: int = 1000
    channels: int = 1
    # daploy
    onnx_path: str = 'model.onnx'

config = Config()
