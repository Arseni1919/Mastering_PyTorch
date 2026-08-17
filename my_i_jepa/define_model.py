import torch
import torch.nn as nn
import torch.nn.functional as F


from config import config
import random
import math
from get_data import sample_context_and_targets
from get_data import dataset


def sigreg_loss(embeddings, num_slices, num_knots=9, t_max=3.5, chunk_slices=None):
    """
    Epps-Pulley statistic against N(0, I), Cramer-Wold sliced.
    Under the null this converges to ~1.06 regardless of N, D, num_slices.
    """
    if embeddings.dim() > 2:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # fp32 throughout: the signal is O(1/sqrt(N)) ~ 0.005, which is
    # the same magnitude as bf16's rounding error near 1.0.
    embeddings = embeddings.float()
    N, D = embeddings.shape
    device = embeddings.device

    A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    z = embeddings @ A                                    # (N, M)

    t = torch.linspace(0.0, t_max, num_knots, device=device, dtype=torch.float32)
    phi = torch.exp(-0.5 * t.square())                    # (K,)

    def _ecf(z_part):
        angle = z_part.unsqueeze(-1) * t.view(1, 1, -1)   # (N, m, K)
        return angle.cos().mean(dim=0), angle.sin().mean(dim=0)

    if chunk_slices is None:
        ecf_real, ecf_imag = _ecf(z)
    else:
        from torch.utils.checkpoint import checkpoint
        reals, imags = [], []
        for zc in z.split(chunk_slices, dim=1):
            r, i = checkpoint(_ecf, zc, use_reentrant=False)
            reals.append(r)
            imags.append(i)
        ecf_real = torch.cat(reals, dim=0)
        ecf_imag = torch.cat(imags, dim=0)

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
        # todo: pos encoding of RoPE
        self.pos_embed = nn.Parameter(torch.rand((1, self.num_patches, embed_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        bs, num_tokens, emb_size = x.shape
        head_size = emb_size // self.num_heads
        h = self.norm1(x)
        q = self.Q(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        k = self.K(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        v = self.V(h).reshape((bs, num_tokens, self.num_heads, head_size)).permute(0, 2, 1, 3)
        # scores: torch.Tensor = (q @ k.transpose(-2, -1)) / (head_size ** 0.5)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        # attn_weights = torch.softmax(scores, dim=-1)
        # h = (attn_weights @ v).permute(0, 2, 1, 3).reshape((bs, num_tokens, emb_size))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 2, 1, 3).reshape((bs, num_tokens, emb_size))
        h = self.out_proj(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            in_channels,
            encoder_embed_dim,
            depth,
            num_heads
    ):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, encoder_embed_dim)
        self.blocks: nn.ModuleList[ViTBlock] = nn.ModuleList()
        for _ in range(depth):
            block = ViTBlock(encoder_embed_dim, num_heads)
            self.blocks.append(block)
        # self.norm = nn.LayerNorm(encoder_embed_dim, elementwise_affine=False)
        self.norm = nn.LayerNorm(encoder_embed_dim)

    def forward(self, x: torch.Tensor, context_mask=None):
        # x: (bs, in_channels, image_size, image_size)
        tokens = self.patch_embed(x)
        if context_mask is not None:
            tokens = tokens[:,context_mask,:]
        for block in self.blocks:
            tokens = block(tokens, mask=None)
        tokens = self.norm(tokens)
        return tokens


class Predictor(nn.Module):
    def __init__(
            self, encoder_embed_dim, predictor_embed_dim, predictor_depth, predictor_num_heads, num_patches
    ):
        super().__init__()
        self.in_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.rand((predictor_embed_dim,)))
        self.pos_embed = nn.Parameter(torch.rand((1, num_patches, predictor_embed_dim)))
        self.blocks = nn.ModuleList()
        for _ in range(predictor_depth):
            block = ViTBlock(predictor_embed_dim, predictor_num_heads)
            self.blocks.append(block)
        self.out_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(self, context_repr: torch.Tensor, target_mask: torch.Tensor):
        # context_repr: (bs, num_context_patches, embed_dim)
        bs, num_context_patches, embed_dim = context_repr.shape
        context_tokens = self.in_proj(context_repr)
        mask_tokens = self.mask_token.reshape((1, 1, -1)) + self.pos_embed[:, target_mask, :]
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
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            predictor_embed_dim,
            predictor_depth,
            predictor_num_heads,
            sigreg_num_slices,
            sigreg_lambda,
            num_target_blocks,
            target_scale_range,
            target_aspect_ratio_range,
            context_scale_range,
            ema_start,
            ema_end
    ):
        super().__init__()
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side ** 2
        self.encoder = Encoder(image_size, patch_size, in_channels, encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.target_encoder = Encoder(
            image_size, patch_size, in_channels, encoder_embed_dim, encoder_depth, encoder_num_heads
        )
        self.predictor = Predictor(encoder_embed_dim, predictor_embed_dim, predictor_depth, predictor_num_heads, num_patches)
        self.sigreg_num_slices =sigreg_num_slices
        self.sigreg_lambda = sigreg_lambda
        self.num_patches_per_side = num_patches_per_side
        self.num_target_blocks = num_target_blocks
        self.target_scale_range = target_scale_range
        self.target_aspect_ratio_range = target_aspect_ratio_range
        self.context_scale_range = context_scale_range
        self.encoder_embed_dim = encoder_embed_dim
        self.ema_momentum = ema_start
        self.ema_momentum_final = ema_end

        self._copy_encoder_weights()
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _copy_encoder_weights(self):
        """Hard-copy context encoder weights into target encoder."""
        for p_c, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.copy_(p_c.data)

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        """
        Exponential Moving Average update: θ_t ← m·θ_t + (1−m)·θ_c

        High momentum (close to 1) means the target encoder changes slowly,
        providing stable prediction targets and preventing collapse.
        momentum is annealed from 0.996 → 1.0 so the target encoder
        freezes completely by the end of training.
        """
        m = momentum if momentum is not None else self.ema_momentum
        for p_c, p_t in zip(self.encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.mul_(m).add_((1.0 - m) * p_c.data)

    def get_current_momentum(self, step: int, total_steps: int) -> float:
        """Linearly anneal EMA momentum from ema_momentum → ema_momentum_final."""
        return self.ema_momentum + (self.ema_momentum_final - self.ema_momentum) * (
                step / max(total_steps - 1, 1)
        )

    def forward(self, x: torch.Tensor):
        context_mask, target_masks = sample_context_and_targets(
            self.num_patches_per_side, self.num_target_blocks, self.target_scale_range,
            self.target_aspect_ratio_range, self.context_scale_range
        )
        context_repr = self.encoder(x, context_mask=context_mask)

        # full = self.encoder(x)  # grad enabled
        # target_repr_full = full.detach()
        target_repr_full = self.target_encoder(x)

        losses = []
        for i in range(self.num_target_blocks):
            prediction = self.predictor(context_repr, target_masks[i])
            target = target_repr_full[:, target_masks[i], :]
            losses.append(F.mse_loss(prediction, target))
        prediction_loss = torch.stack(losses).mean()
        # sigreg = sigreg_loss(full, self.sigreg_num_slices)
        # loss = (1 - self.sigreg_lambda) * prediction_loss + self.sigreg_lambda * sigreg
        sigreg = torch.tensor(0)
        loss = prediction_loss
        return loss, prediction_loss, sigreg


def main():

    # --- SigReg
    # gaussian_samples = torch.randn(512, config.embed_size)
    # loss_gaussian = sigreg_loss(gaussian_samples, config.sigreg_num_slices)
    # print(loss_gaussian)
    # # expected: loss_gaussian is small — real Gaussian samples should barely fail the test
    #
    # collapsed_samples = torch.ones(512, config.embed_size)  # every embedding identical — total collapse
    # loss_collapsed = sigreg_loss(collapsed_samples, config.sigreg_num_slices)
    # print(loss_collapsed)
    # expected: loss_collapsed is large — this is exactly the failure mode SIGReg exists to catch

    # --- predictor
    # predictor = Predictor(
    #     config.encoder_embed_dim,
    #     config.predictor_embed_dim, config.predictor_depth, config.predictor_num_heads,
    #     config.num_patches
    # )
    # context_repr = torch.rand(2, 10, config.encoder_embed_dim)
    # target_indices = torch.Tensor([3, 8, 15, 22]).type(torch.long)
    # predicted = predictor(context_repr, target_indices)
    # print(predicted.shape)
    ## expected: predicted.shape == (2, 4, config.embed_dim)

    # --- encoder
    # encoder = Encoder(
    #     config.image_size, config.patch_size, config.in_channels,
    #     config.encoder_embed_dim, config.encoder_depth, config.encoder_num_heads
    # )
    # num_patches_per_side = int(config.num_patches ** 0.5)
    # context_mask, target_masks = sample_context_and_targets(
    #     num_patches_per_side, 4,
    #     config.target_scale_range, config.target_aspect_ratio_range, config.context_scale_range
    # )
    # x = torch.rand(3, config.in_channels, config.image_size, config.image_size)
    # out = encoder(x, context_mask)
    # print(out.shape)

    # --- i-jepa
    model = IJEPAModel(
        config.image_size,
        config.patch_size,
        config.in_channels,
        config.encoder_embed_dim,
        config.encoder_depth,
        config.encoder_num_heads,
        config.predictor_embed_dim,
        config.predictor_depth,
        config.predictor_num_heads,
        config.sigreg_num_slices,
        config.sigreg_lambda,
        config.num_target_blocks,
        config.target_scale_range,
        config.target_aspect_ratio_range,
        config.context_scale_range,
        config.ema_start,
        config.ema_end
    )
    x, y = dataset[0]
    x = x.unsqueeze(0)
    loss, prediction_loss, sigreg = model(x)
    print(loss.item())



if __name__ == '__main__':
    main()


# class PatchEmbed(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor):
#         return x