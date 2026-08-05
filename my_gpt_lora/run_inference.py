import torch
import torch.nn as nn
from define_lora import LoRALinear, load_model
from config import config
from transformers import GPT2LMHeadModel



def load_lora_weights(
        model_name: str,
        target_modules,
        lora_rank,
        lora_alpha,
        lora_dropout,
        device,
        path='state_dict_lora.pt'):
    base_model, tokenizer = load_model(
        model_name, target_modules, lora_rank, lora_alpha, lora_dropout, device
    )
    state_dict_lora = torch.load(path)
    base_model.load_state_dict(state_dict_lora, strict=False)
    return base_model, tokenizer

def merge_lora(lora_linear: LoRALinear):
    delta_weight = (lora_linear.B @ lora_linear.A) * lora_linear.scaling * 0.01
    if isinstance(lora_linear.base_layer, nn.Linear):
        lora_linear.base_layer.weight.add_(delta_weight)
    else:
        lora_linear.base_layer.weight.add_(delta_weight.T)
    return lora_linear.base_layer


def generate(model, tokenizer, prompt, max_new_tokens):
    ids = tokenizer(prompt, return_tensors='pt').input_ids
    generated = model.generate(ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
    lora_model, tokenizer = load_lora_weights(
        config.model_name, config.target_modules, config.lora_rank, config.lora_alpha, config.lora_dropout, device

    )
    max_new_tokens = 500
    prompt = """First Citizen:"""
    print(generate(base_model, tokenizer, prompt, max_new_tokens)[0])
    print('v -------- L0RA -------- v')
    print(generate(lora_model, tokenizer, prompt, max_new_tokens)[0])


if __name__ == '__main__':
    main()