import torch
import torch.nn as nn
from config import config
import random
import math


def sample_block_mask(num_patches_per_side, scale_range, aspect_ratio_range):
    scale_range_bottom, scale_range_up = scale_range
    scale_ratio = random.random() * (scale_range_up - scale_range_bottom) + scale_range_bottom
    target_area = (num_patches_per_side ** 2) * scale_ratio
    aspect_ratio_range_bottom, aspect_ratio_range_up = aspect_ratio_range
    aspect_ratio = random.random() * (aspect_ratio_range_up - aspect_ratio_range_bottom) + aspect_ratio_range_bottom
    height = min(int(math.sqrt(target_area / aspect_ratio)), num_patches_per_side)
    width = min(int(math.sqrt(target_area * aspect_ratio)), num_patches_per_side)
    corner_h = random.randint(0, num_patches_per_side - height)
    corner_w = random.randint(0, num_patches_per_side - width)
    h_range = torch.arange(corner_h, corner_h + height).reshape((height, 1))
    w_range = torch.arange(corner_w, corner_w + width).unsqueeze(0)
    final_set_of_patch_indices_t = (h_range * num_patches_per_side + w_range).flatten()
    return final_set_of_patch_indices_t


def sample_context_and_targets(
        num_patches_per_side, M, target_scale_range, target_aspect_ratio_range, context_scale_range
):
    target_masks = [
        sample_block_mask(num_patches_per_side, target_scale_range, target_aspect_ratio_range) for _ in range(M)
    ]
    context_mask = sample_block_mask(num_patches_per_side, context_scale_range, aspect_ratio_range=(1.0, 1.0))
    for target_mask in target_masks:
        context_mask = context_mask[torch.isin(context_mask, target_mask, invert=True)]
    return context_mask, target_masks


def sigreg_loss(embeddings, num_slices, num_knots=17, t_max=5.0):
    if embeddings.dim() > 2:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    N, D = embeddings.shape
    device, dtype = embeddings.device, embeddings.dtype
    A = torch.randn(D, num_slices, device=device, dtype=dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    z = embeddings @ A  # (N, M)
    t = torch.linspace(0.0, t_max, num_knots, device=device, dtype=dtype)
    phi = torch.exp(-0.5 * t.square())
    angle = z.unsqueeze(-1) * t.view(1, 1, -1)  # (N, M, K)
    ecf_real = angle.cos().mean(dim=0)  # (M, K)
    ecf_imag = angle.sin().mean(dim=0)  # (M, K)
    sq_err = (ecf_real - phi.view(1, -1)).square() + ecf_imag.square()
    integrand = sq_err * phi.view(1, -1)
    dt = t[1] - t[0]
    w = torch.full_like(t, dt)
    w[0] *= 0.5
    w[-1] *= 0.5
    per_slice = 2.0 * (integrand * w.view(1, -1)).sum(dim=-1)
    return N * per_slice.mean()


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.pos_embed = nn.Parameter(torch.rand((1, self.num_patches, embed_dim)))

    def forward(self, x: torch.Tensor):
        # x: bs, in_ch, image_size, image_size
        # conv(x): bs, emb_s, patch_size, patch_size
        patches = self.conv(x)
        bs, emb_s, patch_size, patch_size = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape((bs, patch_size**2, emb_s))
        return patches + self.pos_embed


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        bs, num_tokens, emb_size = x.shape
        head_size = emb_size // self.num_heads
        h = self.norm1(x)
        q = self.Q(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        k = self.K(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        v = self.V(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(head_size), dim=-1)
        h = (attn_weights @ v).permute(0, 2, 1, 3).reshape((bs, num_tokens, emb_size))
        h = self.out_proj(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, depth, num_heads):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = ViTBlock(embed_dim, num_heads)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor, patch_indices=None):
        # x: (bs, in_channels, image_size, image_size)
        tokens = self.patch_embed(x)
        if patch_indices is not None:
            tokens = tokens[:,patch_indices,:]
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class Predictor(nn.Module):
    def __init__(self, embed_dim, predictor_embed_dim, predictor_depth, predictor_num_heads, num_patches):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.rand((predictor_embed_dim,)))
        self.pos_embed = nn.Parameter(torch.rand((1, num_patches, predictor_embed_dim)))
        self.blocks = nn.ModuleList()
        for _ in range(predictor_depth):
            block = ViTBlock(predictor_embed_dim, predictor_num_heads)
            self.blocks.append(block)
        self.out_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, context_repr: torch.Tensor, target_patch_indices: torch.Tensor):
        # context_repr: (bs, num_context_patches, embed_dim)
        bs, num_context_patches, embed_dim = context_repr.shape
        context_tokens = self.in_proj(context_repr)
        mask_tokens = self.mask_token.reshape((1, 1, -1)) + self.pos_embed[:, target_patch_indices, :]
        mask_tokens = mask_tokens.expand((bs, -1, -1))
        tokens = torch.cat([context_tokens, mask_tokens], dim=1)
        # tokens: (bs, num_context_patches + len(target_patch_indices), predictor_embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        masked_output = tokens[:, num_context_patches:, :]
        predicted = self.out_proj(masked_output)
        # (bs, len(target_patch_indices), embed_dim)
        return predicted



class IJEPAModel(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            encoder_depth,
            encoder_num_heads,
            predictor_embed_dim,
            predictor_depth,
            predictor_num_heads,
            sigreg_num_slices,
            sigreg_lambda,
            M,
            target_scale_range,
            target_aspect_ratio_range,
            context_scale_range
    ):
        super().__init__()
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side ** 2
        self.encoder = Encoder(image_size, patch_size, in_channels, embed_dim, encoder_depth, encoder_num_heads)
        self.predictor = Predictor(embed_dim, predictor_embed_dim, predictor_depth, predictor_num_heads, num_patches)
        self.sigreg_num_slices =sigreg_num_slices
        self.sigreg_lambda = sigreg_lambda
        self.num_patches_per_side = num_patches_per_side
        self.M = M
        self.target_scale_range = target_scale_range
        self.target_aspect_ratio_range = target_aspect_ratio_range
        self.context_scale_range = context_scale_range
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        context_mask, target_masks = sample_context_and_targets(
            self.num_patches_per_side, self.M, self.target_scale_range,
            self.target_aspect_ratio_range, self.context_scale_range
        )
        context_repr = self.encoder(x, patch_indices=context_mask)

        with torch.no_grad():
            target_repr_full = self.encoder(x)

        losses = []
        for i in range(self.M):
            prediction = self.predictor(context_repr, target_masks[i])
            target = target_repr_full[:, target_masks[i], :]
            losses.append(nn.MSELoss()(prediction, target))
        prediction_loss = torch.stack(losses).mean()
        # sigreg = sigreg_loss(self.encoder(x).reshape(-1, self.embed_dim), self.sigreg_num_slices)
        # loss = (1 - self.sigreg_lambda) * prediction_loss + self.sigreg_lambda * sigreg
        sigreg = torch.tensor(0)
        loss = prediction_loss
        return loss, prediction_loss, sigreg


def main():
    gaussian_samples = torch.randn(512, config.embed_size)
    loss_gaussian = sigreg_loss(gaussian_samples, config.sigreg_num_slices)
    print(loss_gaussian)
    # expected: loss_gaussian is small — real Gaussian samples should barely fail the test

    collapsed_samples = torch.ones(512, config.embed_size)  # every embedding identical — total collapse
    loss_collapsed = sigreg_loss(collapsed_samples, config.sigreg_num_slices)
    print(loss_collapsed)
    # expected: loss_collapsed is large — this is exactly the failure mode SIGReg exists to catch
    # predictor = Predictor(32, 128, 3, 4, 49)
    # context_repr = torch.rand(2, 10, config.embed_size)
    # target_indices = [3, 8, 15, 22]
    # predicted = predictor(context_repr, target_indices)
    # print(predicted.shape)
    # expected: predicted.shape == (2, 4, config.embed_dim)
    # encoder = Encoder(28, 4, 1, 32, 3, 4)
    # x = torch.rand(2, 1, 28, 28)
    # full_out = encoder(x)
    # print(full_out.shape)
    # # expected: full_out.shape == (2, 49, config.embed_dim)
    # context_out = encoder(x, patch_indices=list(range(10)))
    # expected: context_out.shape == (2, 10, config.embed_dim)
    # context_mask, target_masks = sample_context_and_targets(
    #     7,
    #     config.num_target_blocks,
    #     config.target_scale_range,
    #     config.target_aspect_ratio_range,
    #     config.context_scale_range
    # )
    # expected: len(target_masks) == config.num_target_blocks
    # expected: context_mask has no indices in common with any target_masks[i]
    # patch_embed = PatchEmbed(28, 7, 1, 32)
    # x = torch.rand(2, 1, 28, 28)
    # tokens = patch_embed(x)
    # print(tokens.shape)


if __name__ == '__main__':
    main()


# class PatchEmbed(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor):
#         return x