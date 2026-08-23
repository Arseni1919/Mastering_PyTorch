from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = 'gpt2'
    prompt: str = 'First Citizen:'
    max_new_tokens: int = 100


config = Config()
