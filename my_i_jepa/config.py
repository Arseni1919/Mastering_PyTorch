from dataclasses import dataclass


@dataclass
class Config:
    patch_size: int = 4
    image_size: int = 28
    in_channels: int = 1
    embed_size: int = 64
    encoder_depth: int = 6
    encoder_num_heads: int = 4
    predictor_embed_dim: int = 32
    predictor_depth: int = 4
    predictor_num_heads: int = 4
    num_target_blocks: int = 4
    target_scale_range: tuple = (0.15, 0.2)
    target_aspect_ratio_range: tuple = (0.75, 1.2)
    context_scale_range: tuple = (0.85, 1.0)
    sigreg_lambda: float = 0.05
    sigreg_num_slices: int = 256
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10


config = Config()