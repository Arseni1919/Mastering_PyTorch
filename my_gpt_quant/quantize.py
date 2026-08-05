import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer
import copy
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from config import config
import  time
import io

from transformers.pytorch_utils import Conv1D

def conv1d_to_linear(conv1d: Conv1D) -> nn.Linear:
    in_features, out_features = conv1d.weight.shape
    linear = nn.Linear(in_features, out_features, bias=conv1d.bias is not None)
    linear.weight.data = conv1d.weight.data.T.contiguous()
    if conv1d.bias is not None:
        linear.bias.data = conv1d.bias.data
    return linear

def replace_conv1d_with_linear(model):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Conv1D):
            parent_name = '.'.join(name.split('.')[:-1])
            attr = name.split('.')[-1]
            setattr(model.get_submodule(parent_name), attr, conv1d_to_linear(mod))
    return model


def load_base_model(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def quantize_model(model: nn.Module):
    model_copy = copy.deepcopy(model)
    replace_conv1d_with_linear(model_copy)
    quantize_(
        model_copy, Int8WeightOnlyConfig(version=2),
        filter_fn=lambda mod, name: isinstance(mod, nn.Linear) and mod is not model_copy.lm_head
    )
    return model_copy


def generate(model, tokenizer, prompt, max_new_tokens):
  ids = tokenizer(prompt, return_tensors='pt').input_ids
  generated = model.generate(ids, max_new_tokens=max_new_tokens)   # no_grad, eval mode
  return tokenizer.decode(generated)[0]


def model_size_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()



def main():
    base_model, tokenizer = load_base_model('gpt2')
    quantized_model = quantize_model(base_model)
    print('--- BASE')
    start = time.time()
    print(generate(base_model, tokenizer, config.prompt, config.max_new_tokens))
    print(f'elapsed time {time.time() - start: .3f}')
    print(f'size: {model_size_bytes(base_model)}')
    print('---')
    print('--- QUANT')
    start = time.time()
    print(generate(quantized_model, tokenizer, config.prompt, config.max_new_tokens))
    print(f'elapsed time {time.time() - start: .3f}')
    print(f'size: {model_size_bytes(quantized_model)}')
    print('---')
    # expected: base_model's weights are still full-precision (untouched) — only quantized_model's are int8


if __name__ == '__main__':
    main()