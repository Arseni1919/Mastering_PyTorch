from dataclasses import dataclass, field

@dataclass
class Config:
    model_name: str = 'gpt2'
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ['c_attn', 'c_proj'])
    block_size: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-4
    n_epochs: int = 1


config = Config()
