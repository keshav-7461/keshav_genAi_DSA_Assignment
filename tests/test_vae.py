"""Tests for the four things the assignment asks about:

1. encoder-decoder architecture
2. latent distribution (sampling + the learned aggregate posterior)
3. reconstruction loss
4. KL-divergence term
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as torch_kl

from backend.data import load_mnist
from backend.train import WEIGHTS_DIR
from backend.vae import (VAE, Decoder, Encoder, kl_divergence,
                         reconstruction_loss, vae_loss)

torch.manual_seed(0)


# ------------------------------------------------------------------ architecture
class TestArchitecture:
    @pytest.mark.parametrize("latent_dim", [2, 8, 32])
    def test_encoder_outputs_mu_and_logvar_of_latent_size(self, latent_dim):
        mu, logvar = Encoder(latent_dim)(torch.rand(5, 784))
        assert mu.shape == (5, latent_dim)
        assert logvar.shape == (5, latent_dim)

    def test_encoder_accepts_28x28_images(self):
        mu, _ = Encoder(2)(torch.rand(3, 28, 28))
        assert mu.shape == (3, 2)

    @pytest.mark.parametrize("latent_dim", [2, 8, 32])
    def test_decoder_maps_latent_back_to_784_pixels(self, latent_dim):
        assert Decoder(latent_dim)(torch.randn(4, latent_dim)).shape == (4, 784)

    def test_decode_output_is_a_valid_pixel_probability(self):
        out = VAE(2).decode(torch.randn(16, 2) * 5)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_full_forward_shapes(self):
        logits, mu, logvar, z = VAE(2)(torch.rand(6, 784))
        assert logits.shape == (6, 784)
        assert mu.shape == logvar.shape == z.shape == (6, 2)

    def test_layer_sizes_form_a_bottleneck(self):
        m = VAE(2)
        widths = [l.out_features for l in m.encoder.body if isinstance(l, torch.nn.Linear)]
        assert widths == [512, 256]
        assert m.encoder.fc_mu.out_features == m.encoder.fc_logvar.out_features == 2
        dwidths = [l.out_features for l in m.decoder.net if isinstance(l, torch.nn.Linear)]
        assert dwidths == [256, 512, 784]

    def test_model_is_small(self):
        assert sum(p.numel() for p in VAE(2).parameters()) < 2_000_000

    def test_gradients_flow_through_the_sampling_step(self):
        """The reparameterisation trick must keep the encoder differentiable."""
        m = VAE(2)
        x = torch.rand(8, 784)
        logits, mu, logvar, _ = m(x)
        vae_loss(logits, x, mu, logvar)[0].backward()
        for name, p in m.encoder.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert m.encoder.fc_logvar.weight.grad.abs().sum() > 0

    def test_eval_forward_is_not_deterministic_but_encode_is(self):
        m = VAE(2).eval()
        x = torch.rand(4, 784)
        assert torch.equal(m.encode(x)[0], m.encode(x)[0])
        assert not torch.equal(m(x)[3], m(x)[3])  # sampled z differs between calls


# ------------------------------------------------------------------ latent distribution
class TestLatentDistribution:
    N = 200_000

    def test_sampling_with_standard_normal_posterior(self):
        z = VAE.sample(torch.zeros(self.N, 2), torch.zeros(self.N, 2))
        assert z.mean(0).abs().max() < 0.02
        assert (z.std(0) - 1).abs().max() < 0.02

    def test_sampling_matches_requested_mean_and_variance(self):
        mu = torch.tensor([3.0, -2.0])
        logvar = torch.log(torch.tensor([4.0, 0.25]))  # sigma = 2, 0.5
        z = VAE.sample(mu.expand(self.N, 2), logvar.expand(self.N, 2))
        assert torch.allclose(z.mean(0), mu, atol=0.03)
        assert torch.allclose(z.std(0), torch.tensor([2.0, 0.5]), atol=0.03)

    def test_zero_variance_posterior_is_deterministic(self):
        mu = torch.randn(5, 2)
        z = VAE.sample(mu, torch.full_like(mu, -50.0))
        assert torch.allclose(z, mu, atol=1e-6)

    def test_sampling_is_reparameterised(self):
        """z must be a differentiable function of mu and logvar."""
        mu = torch.zeros(4, 2, requires_grad=True)
        logvar = torch.zeros(4, 2, requires_grad=True)
        VAE.sample(mu, logvar).sum().backward()
        assert torch.allclose(mu.grad, torch.ones_like(mu))
        assert logvar.grad.abs().sum() > 0

    @pytest.mark.skipif(not (WEIGHTS_DIR / "vae.pt").exists(), reason="no trained weights")
    def test_trained_aggregate_posterior_is_close_to_the_prior(self):
        """After training the encoded test set should roughly look like N(0, I):
        centred, with O(1) spread, and per-image posteriors narrower than the prior."""
        ckpt = torch.load(WEIGHTS_DIR / "vae.pt", map_location="cpu")
        model = VAE(ckpt["latent_dim"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
        x, _ = load_mnist("test")
        with torch.no_grad():
            mu, logvar = model.encode(torch.from_numpy(x).view(-1, 784))
        assert mu.mean(0).abs().max() < 0.35, "latent means should be centred near 0"
        assert 0.5 < mu.std(0).min() and mu.std(0).max() < 2.0, "spread should be O(1)"
        assert logvar.exp().sqrt().mean() < 1.0, "posteriors should be tighter than the prior"


# ------------------------------------------------------------------ reconstruction loss
class TestReconstructionLoss:
    def test_matches_manual_bernoulli_nll(self):
        x = torch.rand(3, 784)
        logits = torch.randn(3, 784)
        p = torch.sigmoid(logits)
        manual = -(x * p.log() + (1 - x) * (1 - p).log()).sum(1).mean()
        assert reconstruction_loss(logits, x).item() == pytest.approx(manual.item(), rel=1e-4)

    def test_is_summed_over_pixels_and_averaged_over_batch(self):
        x = torch.rand(4, 784)
        logits = torch.randn(4, 784)
        per_image = torch.stack([reconstruction_loss(logits[i:i+1], x[i:i+1]) for i in range(4)])
        assert reconstruction_loss(logits, x).item() == pytest.approx(per_image.mean().item(), rel=1e-5)
        # per-pixel mean times 784 gives the same number
        assert reconstruction_loss(logits, x).item() == pytest.approx(
            F.binary_cross_entropy_with_logits(logits, x).item() * 784, rel=1e-5)

    def test_near_zero_for_a_confident_correct_reconstruction(self):
        x = (torch.rand(4, 784) > 0.5).float()
        logits = (x * 2 - 1) * 30.0
        assert reconstruction_loss(logits, x).item() < 1e-3

    def test_better_reconstruction_has_lower_loss(self):
        x = (torch.rand(8, 784) > 0.5).float()
        good = (x * 2 - 1) * 4.0
        bad = -good
        assert reconstruction_loss(good, x) < reconstruction_loss(torch.zeros_like(x), x) < reconstruction_loss(bad, x)

    def test_uninformative_prediction_costs_784_ln2(self):
        x = torch.rand(2, 784)
        assert reconstruction_loss(torch.zeros(2, 784), x).item() == pytest.approx(784 * math.log(2), rel=1e-4)

    def test_never_negative_for_binary_targets(self):
        x = (torch.rand(16, 784) > 0.5).float()
        assert reconstruction_loss(torch.randn(16, 784) * 3, x).item() >= 0

    def test_accepts_28x28_targets(self):
        x = torch.rand(2, 28, 28)
        assert reconstruction_loss(torch.randn(2, 784), x).ndim == 0


# ------------------------------------------------------------------ KL divergence
class TestKLDivergence:
    def test_zero_when_posterior_equals_prior(self):
        assert kl_divergence(torch.zeros(5, 2), torch.zeros(5, 2)).item() == pytest.approx(0.0, abs=1e-7)

    def test_matches_closed_form_for_known_values(self):
        # 1-D: KL(N(mu, s^2) || N(0,1)) = 0.5 * (s^2 + mu^2 - 1 - ln s^2)
        mu, s2 = 1.5, 0.4
        expected = 0.5 * (s2 + mu ** 2 - 1 - math.log(s2))
        got = kl_divergence(torch.tensor([[mu]]), torch.tensor([[math.log(s2)]])).item()
        assert got == pytest.approx(expected, rel=1e-5)

    def test_matches_torch_distributions(self):
        mu, logvar = torch.randn(32, 6), torch.randn(32, 6)
        ref = torch_kl(Normal(mu, (0.5 * logvar).exp()), Normal(torch.zeros(6), torch.ones(6))).sum(1).mean()
        assert kl_divergence(mu, logvar).item() == pytest.approx(ref.item(), rel=1e-4)

    def test_summed_over_latent_dims(self):
        mu, logvar = torch.randn(1, 4), torch.randn(1, 4)
        per_dim = sum(kl_divergence(mu[:, i:i+1], logvar[:, i:i+1]) for i in range(4))
        assert kl_divergence(mu, logvar).item() == pytest.approx(per_dim.item(), rel=1e-5)

    def test_non_negative(self):
        for _ in range(20):
            assert kl_divergence(torch.randn(8, 3) * 3, torch.randn(8, 3) * 2).item() >= -1e-6

    def test_grows_as_the_mean_moves_away_from_the_prior(self):
        lv = torch.zeros(1, 2)
        kls = [kl_divergence(torch.full((1, 2), m), lv).item() for m in (0.0, 1.0, 2.0, 4.0)]
        assert kls == sorted(kls) and kls[0] < kls[-1]

    def test_penalises_variance_collapse_and_blow_up(self):
        mu = torch.zeros(1, 2)
        base = kl_divergence(mu, torch.zeros(1, 2)).item()
        assert kl_divergence(mu, torch.full((1, 2), -4.0)).item() > base
        assert kl_divergence(mu, torch.full((1, 2), 3.0)).item() > base

    def test_total_loss_is_reconstruction_plus_beta_kl(self):
        x = torch.rand(4, 784)
        logits, mu, logvar, _ = VAE(2)(x)
        total, rec, kl = vae_loss(logits, x, mu, logvar, beta=0.5)
        assert total.item() == pytest.approx((rec + 0.5 * kl).item(), rel=1e-6)
        assert vae_loss(logits, x, mu, logvar, beta=0.0)[0].item() == pytest.approx(rec.item(), rel=1e-6)


# ------------------------------------------------------------------ a short training run
def test_a_few_training_steps_reduce_the_loss():
    torch.manual_seed(1)
    x, _ = load_mnist("train")
    x = torch.from_numpy(x[:2048]).view(-1, 784)
    m = VAE(2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def loss_now():
        with torch.no_grad():
            logits, mu, logvar, _ = m(x)
            return vae_loss(logits, x, mu, logvar)[0].item()

    before = loss_now()
    for _ in range(60):
        for i in range(0, len(x), 256):
            xb = x[i:i + 256]
            logits, mu, logvar, _ = m(xb)
            loss = vae_loss(logits, xb, mu, logvar)[0]
            opt.zero_grad(); loss.backward(); opt.step()
    assert loss_now() < before * 0.75
