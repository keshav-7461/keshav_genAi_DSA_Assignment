"""Save static report figures to outputs/:  python -m backend.make_figures"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .app import LatentSpace, load_or_train_model
from .data import load_mnist
from .vae import kl_divergence, reconstruction_loss

OUT = Path(__file__).resolve().parent.parent / "outputs"
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']


def main():
    OUT.mkdir(exist_ok=True)
    model = load_or_train_model()
    space = LatentSpace(model)
    x, y = load_mnist("test")

    # 1. original vs reconstruction: one example of each digit
    picks = [int(np.where(y == d)[0][0]) for d in range(10)]
    xb = torch.from_numpy(x[picks]).view(-1, 784)
    with torch.no_grad():
        mu, logvar = model.encode(xb)
        logits = model.decoder(mu)
        rec = torch.sigmoid(logits).numpy()
    fig, ax = plt.subplots(3, 10, figsize=(12, 4.2))
    for i in range(10):
        ax[0, i].imshow(x[picks[i]], cmap="gray"); ax[1, i].imshow(rec[i].reshape(28, 28), cmap="gray")
        ax[2, i].imshow(np.abs(x[picks[i]] - rec[i].reshape(28, 28)), cmap="magma", vmin=0, vmax=1)
        for r in range(3):
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
    for r, t in enumerate(["Original", "Reconstructed", "|Difference|"]):
        ax[r, 0].set_ylabel(t, fontsize=10)
    fig.suptitle("MNIST test digits: original vs VAE reconstruction", y=0.99)
    fig.tight_layout(); fig.savefig(OUT / "original_vs_reconstructed.png", dpi=150); plt.close(fig)

    # 2. latent space scatter + decoded manifold
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for d in range(10):
        m = space.labels[space.idx] == d
        ax[0].scatter(space.points[m, 0], space.points[m, 1], s=6, c=COLORS[d], label=str(d), alpha=0.75)
    ax[0].set_title(f"Encoded test images (μ), {space.mu.shape[1]}-D latent"
                    + ("" if space.mu.shape[1] == 2 else ", PCA plane"))
    ax[0].set_xlabel("z₁"); ax[0].set_ylabel("z₂"); ax[0].legend(markerscale=2.5, ncol=2, title="digit")
    ax[0].set_aspect("equal")
    n = 20
    lo, hi = space.points.min(0), space.points.max(0)
    xs = np.linspace(lo[0], hi[0], n); ys = np.linspace(hi[1], lo[1], n)
    grid = space.unproject(np.array([[a, b] for b in ys for a in xs], dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        imgs = model.decode(torch.from_numpy(grid)).numpy().reshape(n, n, 28, 28)
    ax[1].imshow(imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28), cmap="gray",
                 extent=[xs[0], xs[-1], ys[-1], ys[0]], aspect="auto")
    ax[1].set_title("Decoded manifold (decoder applied to a grid)")
    ax[1].set_xlabel("z₁"); ax[1].set_ylabel("z₂")
    fig.tight_layout(); fig.savefig(OUT / "latent_space.png", dpi=150); plt.close(fig)

    # 3. training curves
    import json
    h = json.loads((model_dir := Path(__file__).resolve().parent / "weights" / "history.json").read_text())
    e = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, t in zip(ax, ["loss", "rec", "kl"], ["Total loss (neg. ELBO)", "Reconstruction loss (BCE)", "KL divergence"]):
        a.plot(e, [r[f"train_{key}"] for r in h], label="train", color="#4e79a7")
        a.plot(e, [r[f"test_{key}"] for r in h], label="test", color="#e15759", ls="--")
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)

    with torch.no_grad():
        xt = torch.from_numpy(x).view(-1, 784)
        mu, logvar = model.encode(xt)
        r = reconstruction_loss(model.decoder(mu), xt).item(); k = kl_divergence(mu, logvar).item()
    print(f"test set: reconstruction {r:.2f}  KL {k:.2f}  total {r + k:.2f}")
    print("saved:", *[p.name for p in sorted(OUT.glob('*.png'))])


if __name__ == "__main__":
    main()
