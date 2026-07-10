from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).parents[2]))
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
