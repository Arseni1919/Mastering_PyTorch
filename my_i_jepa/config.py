from dataclasses import dataclass


@dataclass
class Config:
    # --- image / patch
    # patch_size: int = 4
    # image_size: int = 28
    # in_channels: int = 1
    patch_size: int = 16
    image_size: int = 224
    in_channels: int = 3
    grid_size = image_size // patch_size
    num_patches = grid_size ** 2
    # --- encoder
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    # --- predictor
    predictor_embed_dim: int = 384
    predictor_depth: int = 6
    predictor_num_heads: int = 12
    # --- masking
    num_target_blocks: int = 4
    target_scale_range: tuple = (0.15, 0.2)
    target_aspect_ratio_range: tuple = (0.75, 1.2)
    context_scale_range: tuple = (0.85, 1.0)
    # --- sigreg
    sigreg_lambda: float = 0.01
    sigreg_num_slices: int = 32
    # --- training
    epochs: int = 70
    batch_size: int = 16
    accum_steps = 4
    lr: float = 1e-3
    lr_min: float = 1e-6
    warmup_epochs = 1
    wd_start = 0.04
    wd_end = 0.4
    # --- ema
    ema_start = 0.996
    ema_end = 1.000




config = Config()