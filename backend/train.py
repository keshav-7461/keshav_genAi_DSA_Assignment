"""Train the VAE on MNIST and save weights to backend/weights/vae.pt.

    python -m backend.train --epochs 20 --latent-dim 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_mnist
from .vae import VAE, vae_loss

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def evaluate(model: VAE, x: torch.Tensor, batch_size: int = 1000, beta: float = 1.0):
    model.eval()
    tot = rec = kl = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size].view(-1, 784)
            logits, mu, logvar, _ = model(xb)
            l, r, k = vae_loss(logits, xb, mu, logvar, beta)
            n = len(xb)
            tot += l.item() * n
            rec += r.item() * n
            kl += k.item() * n
    n = len(x)
    return tot / n, rec / n, kl / n


def train(epochs=20, latent_dim=2, batch_size=128, lr=1e-3, beta=1.0, seed=0, out=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    xtr, _ = load_mnist("train")
    xte, _ = load_mnist("test")
    xtr = torch.from_numpy(xtr).view(-1, 784)
    xte = torch.from_numpy(xte).view(-1, 784)
    loader = DataLoader(TensorDataset(xtr), batch_size=batch_size, shuffle=True)

    model = VAE(latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for (xb,) in loader:
            logits, mu, logvar, _ = model(xb)
            loss, _, _ = vae_loss(logits, xb, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = evaluate(model, xtr, beta=beta)
        te = evaluate(model, xte, beta=beta)
        history.append(
            {"epoch": epoch, "train_loss": tr[0], "train_rec": tr[1], "train_kl": tr[2],
             "test_loss": te[0], "test_rec": te[1], "test_kl": te[2]}
        )
        print(f"epoch {epoch:2d}/{epochs}  train {tr[0]:.2f} (rec {tr[1]:.2f} kl {tr[2]:.2f})  "
              f"test {te[0]:.2f} (rec {te[1]:.2f} kl {te[2]:.2f})  {time.time() - t0:.1f}s", flush=True)

    out = Path(out) if out else WEIGHTS_DIR / "vae.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "latent_dim": latent_dim, "beta": beta}, out)
    (out.parent / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out}")
    return model, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.epochs, a.latent_dim, a.batch_size, a.lr, a.beta, a.seed)
