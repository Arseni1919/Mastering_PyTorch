import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from get_data import tokenizer


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        if isinstance(self.base_layer, nn.Linear):
            out_features, in_features = self.base_layer.weight.shape
        else:
            in_features, out_features = self.base_layer.weight.shape
        self.A = nn.Parameter(torch.rand((r, in_features)))
        self.B = nn.Parameter(torch.zeros((out_features, r)))
        # self.register_buffer('A', torch.rand((r, in_features)))
        # self.register_buffer('B', torch.zeros((out_features, r)))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor):
        base_out = self.base_layer(x)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T
        return base_out + lora_out * self.scaling



def inject_lora(model: nn.Module, target_modules, r, alpha, dropout):
    for p in model.parameters():
        p.requires_grad = False
    for mod_name, mod in model.named_modules():
        name_list = mod_name.split('.')
        parent_name = '.'.join(name_list[:-1])
        if name_list[-1] in target_modules:
            new_mod = LoRALinear(mod, r, alpha, dropout)
            setattr(model.get_submodule(parent_name), name_list[-1], new_mod)
    return model


def load_model(model_name, target_modules, r, alpha, dropout, device):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = inject_lora(model, target_modules, r, alpha, dropout)
    model.to(device)
    return model, tokenizer




def main():
    model, tokenizer = load_model('gpt2', ['c_attn', 'c_proj'], r=8, alpha=16, dropout=0.05, device='cpu')
    ids = tokenizer('To be or not', return_tensors='pt').input_ids
    out = model(ids).last_hidden_state
    print(out.shape)
    # expected: out.shape == (1, ids.shape[1], vocab_size)
    # base = nn.Linear(32, 16)
    # lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
    # x = torch.rand(2, 32)
    # out = lora(x)
    # print(out.shape)
    # print(torch.allclose(out, base(x)))


if __name__ == '__main__':
    main()


# class LoRALinear(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor):
#         return x