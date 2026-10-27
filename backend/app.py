"""Flask API + static frontend for the VAE latent-space visualizer.

    python -m backend.app            # http://127.0.0.1:5000
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from .data import load_mnist
from .train import WEIGHTS_DIR, train
from .vae import VAE, kl_divergence, reconstruction_loss

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
WEIGHTS_PATH = WEIGHTS_DIR / "vae.pt"


def load_or_train_model(path: Path = WEIGHTS_PATH, epochs: int = 10) -> VAE:
    """Load saved weights, or train a small VAE if none exist yet."""
    if not path.exists():
        print(f"No weights at {path}; training a small VAE for {epochs} epochs...")
        train(epochs=epochs, out=path)
    ckpt = torch.load(path, map_location="cpu")
    model = VAE(ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class LatentSpace:
    """Caches the encoded test set and a 2-D projection used for plotting."""

    def __init__(self, model: VAE, n_points: int = 4000, seed: int = 0):
        self.model = model
        images, labels = load_mnist("test")
        self.images, self.labels = images, labels
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(len(images), size=min(n_points, len(images)), replace=False)
        with torch.no_grad():
            mu, logvar = model.encode(torch.from_numpy(images[self.idx]).view(-1, 784))
        self.mu = mu.numpy()
        self.logvar = logvar.numpy()
        d = model.latent_dim
        if d == 2:
            self.mean = np.zeros(2, dtype=np.float32)
            self.components = np.eye(2, dtype=np.float32)
        else:  # PCA of the posterior means
            self.mean = self.mu.mean(axis=0)
            _, _, vt = np.linalg.svd(self.mu - self.mean, full_matrices=False)
            self.components = vt[:2].astype(np.float32)
        self.points = (self.mu - self.mean) @ self.components.T

    def project(self, mu: np.ndarray) -> np.ndarray:
        return (mu - self.mean) @ self.components.T

    def unproject(self, p: np.ndarray) -> np.ndarray:
        return self.mean + p @ self.components


def create_app(model: VAE | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    model = model or load_or_train_model()
    space = LatentSpace(model)
    app.config["MODEL"] = model
    app.config["SPACE"] = space

    def parse_pixels(payload) -> torch.Tensor:
        px = payload.get("pixels") if isinstance(payload, dict) else None
        if not isinstance(px, list) or len(px) != 784:
            raise ValueError("'pixels' must be a list of 784 numbers in [0, 1]")
        arr = np.asarray(px, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("'pixels' contains non-finite values")
        return torch.from_numpy(np.clip(arr, 0.0, 1.0)).view(1, 784)

    @app.errorhandler(ValueError)
    def bad_request(err):
        return jsonify(error=str(err)), 400

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(FRONTEND_DIR, name)

    @app.get("/api/info")
    def info():
        n_params = sum(p.numel() for p in model.parameters())
        return jsonify(latent_dim=model.latent_dim, n_params=n_params, n_points=len(space.idx),
                       projection="identity" if model.latent_dim == 2 else "pca")

    @app.get("/api/latent")
    def latent():
        return jsonify(
            points=space.points.round(4).tolist(),
            labels=space.labels[space.idx].tolist(),
            indices=space.idx.tolist(),
        )

    @app.get("/api/test_image")
    def test_image():
        if "index" in request.args:
            i = int(request.args["index"])
            if not 0 <= i < len(space.images):
                raise ValueError("index out of range")
        else:
            rng = np.random.default_rng()
            if "digit" in request.args:
                d = int(request.args["digit"])
                if not 0 <= d <= 9:
                    raise ValueError("digit must be 0-9")
                i = int(rng.choice(np.where(space.labels == d)[0]))
            else:
                i = int(rng.integers(len(space.images)))
        return jsonify(index=i, label=int(space.labels[i]),
                       pixels=space.images[i].reshape(-1).round(4).tolist())

    @app.post("/api/reconstruct")
    def reconstruct():
        payload = request.get_json(silent=True) or {}
        x = parse_pixels(payload)
        use_sample = bool(payload.get("sample", False))
        with torch.no_grad():
            mu, logvar = model.encode(x)
            z = model.sample(mu, logvar) if use_sample else mu
            logits = model.decoder(z)
            recon = torch.sigmoid(logits)
            rec_loss = reconstruction_loss(logits, x).item()
            kl = kl_divergence(mu, logvar).item()
        mu_np, lv_np = mu[0].numpy(), logvar[0].numpy()
        return jsonify(
            mu=mu_np.round(4).tolist(),
            sigma=np.exp(0.5 * lv_np).round(4).tolist(),
            z=z[0].numpy().round(4).tolist(),
            point=space.project(mu_np[None])[0].round(4).tolist(),
            reconstruction=recon[0].numpy().round(4).tolist(),
            reconstruction_loss=rec_loss,
            kl_divergence=kl,
            elbo_loss=rec_loss + kl,
        )

    @app.post("/api/decode")
    def decode():
        payload = request.get_json(silent=True) or {}
        p = payload.get("point")
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(v, (int, float)) for v in p):
            raise ValueError("'point' must be [x, y]")
        z = space.unproject(np.asarray(p, dtype=np.float32)[None])
        with torch.no_grad():
            img = model.decode(torch.from_numpy(z.astype(np.float32)))
        return jsonify(image=img[0].numpy().round(4).tolist(), z=z[0].round(4).tolist())

    @app.get("/api/manifold")
    def manifold():
        n = max(2, min(int(request.args.get("n", 15)), 30))
        lo, hi = space.points.min(axis=0), space.points.max(axis=0)
        pad = 0.05 * (hi - lo)
        xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], n)
        ys = np.linspace(hi[1] + pad[1], lo[1] - pad[1], n)  # top row = max y
        grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
        z = space.unproject(grid).astype(np.float32)
        with torch.no_grad():
            imgs = model.decode(torch.from_numpy(z)).numpy()
        return jsonify(n=n, images=imgs.round(3).tolist(),
                       extent=[float(xs[0]), float(xs[-1]), float(ys[-1]), float(ys[0])])

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="127.0.0.1", port=port, debug=False)
