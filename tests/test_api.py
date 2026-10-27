"""API tests. They use an untrained model so they are fast and need no weights file."""
import math

import numpy as np
import pytest
import torch

from backend.app import create_app
from backend.vae import VAE


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    app = create_app(model=VAE(2).eval())
    app.testing = True
    return app.test_client()


def test_index_serves_the_frontend(client):
    r = client.get("/")
    assert r.status_code == 200 and b"VAE Latent Space Visualizer" in r.data


def test_info(client):
    d = client.get("/api/info").get_json()
    assert d["latent_dim"] == 2 and d["n_params"] > 0


def test_latent_returns_labelled_2d_points(client):
    d = client.get("/api/latent").get_json()
    assert len(d["points"]) == len(d["labels"]) == len(d["indices"]) > 1000
    assert all(len(p) == 2 for p in d["points"][:50])
    assert set(d["labels"]) == set(range(10))


def test_reconstruct_returns_everything_the_frontend_needs(client):
    img = client.get("/api/test_image?digit=3").get_json()
    assert img["label"] == 3 and len(img["pixels"]) == 784
    r = client.post("/api/reconstruct", json={"pixels": img["pixels"]}).get_json()
    assert len(r["reconstruction"]) == 784
    assert all(0.0 <= v <= 1.0 for v in r["reconstruction"])
    assert len(r["mu"]) == len(r["sigma"]) == len(r["z"]) == len(r["point"]) == 2
    assert r["elbo_loss"] == pytest.approx(r["reconstruction_loss"] + r["kl_divergence"], rel=1e-4)
    assert r["kl_divergence"] >= 0 and r["reconstruction_loss"] > 0


def test_reconstruct_losses_agree_with_a_manual_computation(client):
    px = client.get("/api/test_image?index=5").get_json()["pixels"]
    r = client.post("/api/reconstruct", json={"pixels": px}).get_json()
    x, p = np.array(px), np.clip(np.array(r["reconstruction"]), 1e-6, 1 - 1e-6)
    manual = -(x * np.log(p) + (1 - x) * np.log(1 - p)).sum()
    assert r["reconstruction_loss"] == pytest.approx(manual, rel=0.01)
    mu, s = np.array(r["mu"]), np.array(r["sigma"])
    kl = 0.5 * (s ** 2 + mu ** 2 - 1 - 2 * np.log(s)).sum()
    assert r["kl_divergence"] == pytest.approx(kl, rel=0.01)


def test_mean_decoding_is_deterministic_and_sampling_is_not(client):
    px = client.get("/api/test_image?index=7").get_json()["pixels"]
    a = client.post("/api/reconstruct", json={"pixels": px}).get_json()
    b = client.post("/api/reconstruct", json={"pixels": px}).get_json()
    assert a["reconstruction"] == b["reconstruction"] and a["z"] == a["mu"]
    s1 = client.post("/api/reconstruct", json={"pixels": px, "sample": True}).get_json()
    s2 = client.post("/api/reconstruct", json={"pixels": px, "sample": True}).get_json()
    assert s1["z"] != s2["z"] and s1["mu"] == s2["mu"]


@pytest.mark.parametrize("payload", [
    {}, {"pixels": [0.0] * 10}, {"pixels": "abc"}, {"pixels": [float("nan")] * 784},
])
def test_reconstruct_rejects_bad_input(client, payload):
    r = client.post("/api/reconstruct", json=payload)
    assert r.status_code == 400 and "error" in r.get_json()


def test_out_of_range_pixels_are_clipped_not_fatal(client):
    r = client.post("/api/reconstruct", json={"pixels": [5.0] * 392 + [-3.0] * 392})
    assert r.status_code == 200


def test_decode_a_latent_point(client):
    d = client.post("/api/decode", json={"point": [0.5, -0.5]}).get_json()
    assert len(d["image"]) == 784 and all(0 <= v <= 1 for v in d["image"])
    assert client.post("/api/decode", json={"point": [1]}).status_code == 400


def test_manifold_grid(client):
    d = client.get("/api/manifold?n=6").get_json()
    assert d["n"] == 6 and len(d["images"]) == 36 and len(d["images"][0]) == 784


def test_test_image_validation(client):
    assert client.get("/api/test_image?index=999999").status_code == 400
    assert client.get("/api/test_image?digit=11").status_code == 400
