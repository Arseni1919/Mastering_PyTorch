from pathlib import Path
import torch
from utils import load_impl, seed_everything


PROJECT = Path(__file__).parents[1]


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
