from dataclasses import dataclass


@dataclass
class Config:
    num_frames: int = 128
    side_size: int = 64
    channels: int = 1
    vae_patch_size: int = 2
    num_downsample_stages: int = 3
    num_temporal_downsample_stages: int = 2
    vae_base_channels: int = 32
    vae_latent_channels: int = 16
    temporal_compression: int = 4
    spatial_compression: int = 8
    vae_kl_weight: float = 1e-5
    vae_dwt_weight: float = 0.1
    vae_perceptual_weight: float = 0.1
    vae_gan_weight: float = 0.05
    vae_denoise_t_max: float = 0.2
    vae_blocks_per_stage: int = 2
    num_classes: int = 6
    class_emb_size: int = 8
    hidden_size: int = 96
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4
    n_epochs_vae: int = 20
    n_epochs_dit: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    disc_learning_rate: float = 1e-4
    num_inference_steps: int = 20
    image_cond_prob: float = 0.5
    vae_model_name: str = 'saved_vae_model'


config = Config()