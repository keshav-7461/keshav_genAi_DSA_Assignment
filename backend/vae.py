"""A small convolution-free VAE for MNIST.

Architecture
------------
Encoder : 784 -> 512 -> 256 -> (mu, logvar), each of size ``latent_dim``
Sampler : z = mu + sigma * eps,  eps ~ N(0, I)   (reparameterisation trick)
Decoder : latent_dim -> 256 -> 512 -> 784 (sigmoid)

Loss (negative ELBO, summed over pixels, averaged over the batch)
-----------------------------------------------------------------
reconstruction = BCE(x_hat, x)                      (sum over 784 pixels)
KL             = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
total          = reconstruction + beta * KL
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 28 * 28


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden=(512, 256)):
        super().__init__()
        h1, h2 = hidden
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.body(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden=(256, 512)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, INPUT_DIM),
        )

    def forward(self, z: torch.Tensor):
        """Returns logits; apply sigmoid to get pixel probabilities."""
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + sigma * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        return self.decoder(z), mu, logvar, z  # logits, mu, logvar, z


def reconstruction_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Bernoulli negative log-likelihood, summed over pixels, mean over batch."""
    x = x.view(x.size(0), -1)
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims, mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def vae_loss(logits, x, mu, logvar, beta: float = 1.0):
    rec = reconstruction_loss(logits, x)
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl
