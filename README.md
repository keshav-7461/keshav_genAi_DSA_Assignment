# VAE Latent Space Visualizer

Encode MNIST digits into a 2-D latent space with a small Variational Autoencoder, decode them back,
and explore the latent space in the browser.

## Run it

```bash
pip install -r requirements.txt

# 1. MNIST: the four *.gz IDX files go in data/raw/ (already included if you got this folder zipped)
#    train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz
#    t10k-images-idx3-ubyte.gz   t10k-labels-idx1-ubyte.gz

python -m backend.train --epochs 20      # optional: weights are already in backend/weights/vae.pt
python -m backend.app                    # open http://127.0.0.1:5000
python -m pytest -q                      # run the tests
python -m backend.make_figures           # regenerate the report figures in outputs/
```

If `backend/weights/vae.pt` is missing, the server trains a small model automatically on first start.

## Layout

| Path | What it is |
|---|---|
| `backend/vae.py` | Encoder, reparameterised sampling, decoder, reconstruction loss, KL term |
| `backend/train.py` | Training loop (Adam, batch 128), saves weights and `history.json` |
| `backend/app.py` | Flask API + serves the frontend |
| `frontend/index.html` | Input (test digit / draw / upload), original vs reconstruction, latent scatter plot |
| `tests/` | 48 tests: architecture, latent distribution, reconstruction loss, KL, API |
| `outputs/` | Static figures: original vs reconstructed, latent space + manifold, training curves |

## Model

```
Encoder : 784 -> 512 -> 256 -> mu (2), logvar (2)
Sampler : z = mu + exp(0.5 * logvar) * eps,   eps ~ N(0, I)
Decoder : 2 -> 256 -> 512 -> 784 (sigmoid)
Loss    : BCE(x_hat, x) summed over pixels  +  KL(N(mu, sigma^2) || N(0, I))
```

KL term: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, summed over latent dimensions, averaged over the batch.

Result after 20 epochs (MNIST test set): reconstruction 132.0, KL 6.8, total loss (negative ELBO) 138.9.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/info` | latent size, parameter count |
| `GET /api/latent` | encoded means + labels for 4000 test images (the scatter plot) |
| `GET /api/test_image?digit=7` | a random test image (or `?index=N`) |
| `POST /api/reconstruct` | `{pixels: [784], sample: bool}` -> mu, sigma, z, reconstruction, recon loss, KL |
| `POST /api/decode` | `{point: [x, y]}` -> decoded image |
| `GET /api/manifold?n=20` | decoder applied to a grid over the latent plane |

## Using a bigger latent space

`python -m backend.train --latent-dim 16` works too. The plot then shows the first two principal
components of the encoded means, and clicks decode the matching point in that plane.
