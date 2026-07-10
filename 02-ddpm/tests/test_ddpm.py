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


def test_unet_forward_shape():
    net = load_impl(PROJECT).UNet()
    out = net(torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)), torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32)


def test_unet_param_count_in_expected_range():
    n = sum(p.numel() for p in load_impl(PROJECT).UNet().parameters())
    assert 30e6 < n < 45e6, n


def test_unet_accepts_null_label():
    net = load_impl(PROJECT).UNet(n_classes=10)
    out = net(torch.randn(1, 3, 32, 32), torch.zeros(1).long(), torch.full((1,), 10))
    assert out.shape == (1, 3, 32, 32)


def test_unet_is_conditional_on_label_and_time():
    impl = load_impl(PROJECT)
    seed_everything(0)
    net = impl.UNet()
    x, t = torch.randn(1, 3, 32, 32), torch.zeros(1).long()
    a = net(x, t, torch.zeros(1).long())
    assert not torch.allclose(a, net(x, t, torch.ones(1).long()))
    assert not torch.allclose(a, net(x, torch.full((1,), 500), torch.zeros(1).long()))


def test_q_sample_matches_closed_form():
    impl = load_impl(PROJECT)
    diff = impl.Diffusion(impl.UNet())
    x0, noise = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    t = torch.tensor([0, 1, 500, 999])
    a = diff.abar[t][:, None, None, None]
    assert torch.allclose(diff.q_sample(x0, t, noise), a.sqrt() * x0 + (1 - a).sqrt() * noise)


def test_q_sample_endpoints_are_data_and_noise():
    impl = load_impl(PROJECT)
    diff = impl.Diffusion(impl.UNet())
    x0, noise = torch.randn(1, 3, 32, 32), torch.ones(1, 3, 32, 32)
    assert torch.allclose(diff.q_sample(x0, torch.zeros(1).long(), noise), x0, atol=1e-2)
    assert torch.allclose(diff.q_sample(x0, torch.full((1,), 999), noise), noise, atol=1e-2)


def test_p_losses_is_scalar_and_finite():
    impl = load_impl(PROJECT)
    diff = impl.Diffusion(impl.UNet())
    loss = diff.p_losses(torch.randn(2, 3, 32, 32), torch.zeros(2).long())
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_p_losses_drops_labels_to_null_token():
    impl = load_impl(PROJECT)
    diff = impl.Diffusion(impl.UNet())
    seen = set()
    original = diff.model.label_emb.forward
    diff.model.label_emb.forward = lambda y: seen.update(y.tolist()) or original(y)
    for _ in range(40):
        diff.p_losses(torch.randn(8, 3, 32, 32), torch.zeros(8).long(), p_drop=0.5)
    assert 10 in seen


def test_eps_with_w_one_equals_conditional():
    impl = load_impl(PROJECT)
    seed_everything(0)
    diff = impl.Diffusion(impl.UNet()).eval()
    x, t, y = torch.randn(2, 3, 32, 32), torch.zeros(2).long(), torch.ones(2).long()
    with torch.no_grad():
        assert torch.allclose(diff.eps(x, t, y, w=1.0), diff.model(x, t, y), atol=1e-5)


def test_samplers_return_correct_shape():
    impl = load_impl(PROJECT)
    diff = impl.Diffusion(impl.UNet(ch=32, mults=(1, 2)), T=20).eval()
    y = torch.arange(2)
    with torch.no_grad():
        assert diff.ddim_sample(y, steps=4).shape == (2, 3, 32, 32)
        assert diff.ddpm_sample(y).shape == (2, 3, 32, 32)


def test_overfits_a_single_batch():
    impl = load_impl(PROJECT)
    seed_everything(0)
    diff = impl.Diffusion(impl.UNet(ch=32, mults=(1, 2)), T=100)
    opt = torch.optim.AdamW(diff.parameters(), lr=2e-3)
    x0, y = torch.randn(4, 3, 32, 32), torch.arange(4) % 10
    losses = []
    for _ in range(400):
        loss = diff.p_losses(x0, y, p_drop=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:20]) / 20
