from dataclasses import dataclass


@dataclass
class Config:
    num_classes: int = 10
    class_emb_size: int = 8
    time_emb_size: int = 8
    patch_size: int = 4
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 6
    mlp_ratio: int = 4
    side_size: int = 32
    n_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_inference_steps: int = 50
    channels: int = 1
    onnx_path: str = 'model.onnx'


config = Config()
