import math
import torch.nn as nn
import torch
from config import config


def sinusoidal_embedding(t: torch.Tensor, dim):
     # if t < 1:
     #    print()
     half = dim // 2
     freqs = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(half) / half).to(t.device)
     args = torch.unsqueeze(t, -1) * freqs
     emb = torch.concat([args.sin(), args.cos()], dim=-1)
     return emb


class ResNetBlock(nn.Module):
    # resnet_block(x, t_emb):
    #   h = normalize(x) -> nonlinearity -> spatial_conv(h)
    #   h = h + project(t_emb) broadcast over spatial dims   # inject time
    #   h = normalize(h) -> nonlinearity -> spatial_conv(h)
    #   return h + (x, or a channel-matching projection of x if channels changed)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(num_groups=config.num_groups, num_channels=in_channels)
        self.group_norm2 = nn.GroupNorm(num_groups=config.num_groups, num_channels=out_channels)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.project = nn.Linear(config.time_emb_size, out_channels)
        self.conv3 = None
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


    def forward(self, x, t_emb):
        h = self.conv1(self.silu(self.group_norm1(x)))
        t_emb = self.project(t_emb)
        h += t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        h = self.conv2(self.silu(self.group_norm2(h)))
        if self.conv3 is not None:
            x = self.conv3(x)
        return h + x


class AttnBlock(nn.Module):
    """
    attn_block(x):
      h = normalize(x)
      q, k, v = project(h) each, per-pixel (no mixing across pixels)
      reshape q, k, v so spatial positions become a sequence
      attention_weights = softmax( (q @ k^T) / sqrt(head_dim) )   # over spatial positions
      h = attention_weights @ v
      reshape h back to spatial layout
      return x + project_out(h)
    """
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.Q = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.K = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.V = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)

    def forward(self, x):
        b_size, ch, height, width = x.shape
        h = self.group_norm(x)
        q: torch.Tensor = self.Q(h).permute(0, 2, 3, 1).reshape(b_size, height*width, ch)
        k: torch.Tensor = self.K(h).permute(0, 2, 3, 1).reshape(b_size, height*width, ch)
        v: torch.Tensor = self.V(h).permute(0, 2, 3, 1).reshape(b_size, height*width, ch)
        head_dim = k.shape[-1]
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)
        h = attn_weights @ v
        h = h.reshape(b_size, height, width, ch).permute(0, 3, 1, 2)
        proj_out = self.out(h)
        return x + proj_out


class DownBlock(nn.Module):
    """
    down_block(x, t_emb, has_attn):
      x = resnet_block(x, t_emb)
      if has_attn: x = attn_block(x)
      skip = x                      # SAVE this for the matching up block
      x = downsample(x)             # strided conv or avgpool, halves H,W
      return x, skip
    """
    def __init__(self, in_channels, out_channels, has_attn, num_groups):
        super().__init__()
        self.resnet_block = ResNetBlock(in_channels, out_channels)
        self.attn_block = None
        if has_attn:
            self.attn_block = AttnBlock(num_groups=num_groups, num_channels=out_channels)
        # self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.resnet_block(x, t_emb)
        if self.attn_block is not None:
            x = self.attn_block(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class MidBlock(nn.Module):
    """
    mid_block(x, t_emb):
      x = resnet_block(x, t_emb)
      x = attn_block(x)
      x = resnet_block(x, t_emb)
      return x
    """
    def __init__(self, num_channels, num_groups):
        super().__init__()
        self.resnet_block1 = ResNetBlock(num_channels, num_channels)
        self.attn_block = AttnBlock(num_groups=num_groups, num_channels=num_channels)
        self.resnet_block2 = ResNetBlock(num_channels, num_channels)

    def forward(self, x, t_emb):
        x = self.resnet_block1(x, t_emb)
        x = self.attn_block(x)
        x = self.resnet_block2(x, t_emb)
        return x


class UpBlock(nn.Module):
    """
    up_block(x, skip, t_emb, has_attn):
      x = upsample(x)                    # nearest/transpose conv, doubles H,W
      x = concat([x, skip], dim=channel) # this is the U-Net magic
      x = resnet_block(x, t_emb)
      if has_attn: x = attn_block(x)
      return x
    """
    def __init__(self, in_channels, out_channels, has_attn, num_groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.resnet_block = ResNetBlock(in_channels * 2, out_channels)
        self.attn_block = None
        if has_attn:
            self.attn_block = AttnBlock(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resnet_block(x, t_emb)
        if self.attn_block is not None:
            x = self.attn_block(x)
        return x


class MyUNetClassConditionedModel(nn.Module):

    def __init__(self, example_x):
        super().__init__()
        ch, height, width = example_x.shape
        base_ch = config.base_channels
        has_attn = config.has_attn
        self.class_embedding = nn.Embedding(config.num_classes, config.class_emb_size)
        time_emd_size = config.time_emb_size
        self.time_embedding = nn.Sequential(
            nn.Linear(time_emd_size, 4 * time_emd_size),
            nn.SiLU(),
            nn.Linear(4 * time_emd_size, time_emd_size)
        )
        self.conv_in = nn.Conv2d(
            in_channels=ch + config.class_emb_size, out_channels=base_ch, kernel_size=3, padding=1
        )
        self.conv_out = nn.Conv2d(
            in_channels=base_ch, out_channels=ch, kernel_size=3, padding=1
        )
        self.down_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            in_channels = base_ch * 2**i
            out_channels = base_ch * 2**(i+1)
            down_block = DownBlock(in_channels=in_channels, out_channels=out_channels, has_attn=has_attn, num_groups=config.num_groups)
            self.down_blocks.append(down_block)
        # self.down_blocks = nn.ModuleList([
        #     DownBlock(in_channels=base_ch, out_channels=base_ch*2, has_attn=has_attn, num_groups=4),
        #     DownBlock(in_channels=base_ch*2, out_channels=base_ch*4, has_attn=has_attn, num_groups=4),
        # ])
        num_channels = base_ch * (2**config.num_layers)
        self.mid_block = MidBlock(num_channels=num_channels, num_groups=config.num_groups)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(config.num_layers)):
            in_channels = base_ch * 2**(i+1)
            out_channels = base_ch * 2**i
            up_block = UpBlock(in_channels=in_channels, out_channels=out_channels, has_attn=has_attn, num_groups=config.num_groups)
            self.up_blocks.append(up_block)
        # self.up_blocks = nn.ModuleList([
        #     UpBlock(in_channels=base_ch*4, out_channels=base_ch*2, has_attn=has_attn, num_groups=4),
        #     UpBlock(in_channels=base_ch*2, out_channels=base_ch, has_attn=has_attn, num_groups=4),
        # ])


    def forward(self, x, t, class_labels):
        """
        x = concat(x, class_cond)              # already implemented
        t_emb = time_embed(t)                  # already implemented
        x = conv_in(x)                         # project to base_channels
        skips = []
        for each down stage (channels doubling, e.g. base -> base*2 -> base*4):
          x, s = down_block(x, t_emb, has_attn=<only at lower resolutions>)
          skips.append(s)
        x = mid_block(x, t_emb)
        for each up stage (mirrored, channels halving back down):
          x = up_block(x, skips.pop(), t_emb, has_attn=<matching the down side>)
        return conv_out(x)                     # project back to original image channels
        """
        bs, channels, h, w = x.shape
        # to add class conditioning
        class_cond = self.class_embedding(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], h, w)
        x = torch.cat([x, class_cond], dim=1)
        # to add time conditioning
        t_emb = sinusoidal_embedding(t , config.time_emb_size)
        t_emb = self.time_embedding(t_emb)
        # main
        x = self.conv_in(x)
        skips = []
        for down_block in self.down_blocks:
            x, s = down_block(x, t_emb)
            skips.append(s)
        x = self.mid_block(x, t_emb)
        for up_block in self.up_blocks:
            x = up_block(x, skips.pop(), t_emb)
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    bs = 3
    net = MyUNetClassConditionedModel(torch.zeros(1, 24, 24))
    net(torch.zeros(bs, 1, 24, 24), torch.randint(100, (bs,)),  torch.randint(10, (bs,)))
