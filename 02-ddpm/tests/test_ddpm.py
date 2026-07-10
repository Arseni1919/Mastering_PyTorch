from pathlib import Path
import torch
from utils import load_impl, seed_everything


PROJECT = Path(__file__).parents[1]


def nn_init_zero(module):
    torch.nn.init.zeros_(module.weight)
    torch.nn.init.zeros_(module.bias)


def test_load_impl_exposes_public_surface():
    impl = load_impl(PROJECT)
    for name in ["UNet", "Diffusion", "cosine_abar", "timestep_embedding"]:
        assert hasattr(impl, name), name


def test_seed_everything_is_deterministic():
    seed_everything(0)
    a = torch.randn(4)
    seed_everything(0)
    assert torch.equal(a, torch.randn(4))


def test_cosine_abar_properties():
    abar = load_impl(PROJECT).cosine_abar(1000)
    assert abar.shape == (1000,)
    assert abar[0] > 0.999
    assert abar[-1] < 0.01
    assert torch.all(abar[1:] < abar[:-1])


def test_timestep_embedding_shape_and_range():
    emb = load_impl(PROJECT).timestep_embedding(torch.arange(8), 128)
    assert emb.shape == (8, 128)
    assert emb.abs().max() <= 1.0


def test_resblock_shape_and_residual_identity():
    impl = load_impl(PROJECT)
    seed_everything(0)
    block = impl.ResBlock(64, 64, 128)
    nn_init_zero(block.conv2)
    x, emb = torch.randn(2, 64, 16, 16), torch.randn(2, 128)
    assert torch.allclose(block(x, emb), x, atol=1e-6)


def test_resblock_changes_channels():
    block = load_impl(PROJECT).ResBlock(64, 128, 128)
    out = block(torch.randn(2, 64, 16, 16), torch.randn(2, 128))
    assert out.shape == (2, 128, 16, 16)


def test_attention_is_residual_and_shape_preserving():
    impl = load_impl(PROJECT)
    attn = impl.Attention(64)
    nn_init_zero(attn.proj)
    x = torch.randn(2, 64, 16, 16)
    assert torch.allclose(attn(x), x, atol=1e-6)


def test_sampling_ops_halve_and_double():
    impl = load_impl(PROJECT)
    x = torch.randn(2, 64, 16, 16)
    assert impl.Downsample(64)(x).shape == (2, 64, 8, 8)
    assert impl.Upsample(64)(x).shape == (2, 64, 32, 32)
