import torch.nn as nn
import torch
from config import config
from get_data import dataset


def sinusoidal_embedding(t, dim):
     half = dim // 2
     freqs = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(half) / half)
     args = t * freqs
     emb = torch.concat([args.sin(), args.cos()], dim=-1)
     return emb


class MyUNetClassConditionedModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.time_embed = nn.Sequential(
           nn.Linear(dim, 4 * dim),
           nn.SiLU(),
           nn.Linear(4 * dim, dim)
        )

    def resnet_block(self, t_emb):
        pass

    def attn_block(self, x):
        pass

    def down_block(self, x, t_emb, has_attn):
        pass

    def middle_block(self, x, t_emb):
        pass

    def up_block(self, x, t_emb, has_attn):
        pass

    def forward(self, x, t, class_labels):
        t_emb = sinusoidal_embedding(t, self.dim)
        t_emb = self.time_embed(t_emb)
        class_cond = self.class_embedding(class_labels)

        # skips = []
        # for block in self.down_blocks:
        #     x, s = block(x, t_emb)
        #     skips.append(s)
        #
        # x = self.middle_block(x, t_emb)
        #
        # for block in self.up_blocks:
        #     x = block(x, skips.pop(), t_emb)

        return x


if __name__ == '__main__':
    x, y = dataset[0]
    c, w, h = x.shape
    net = MyUNetClassConditionedModel(w)