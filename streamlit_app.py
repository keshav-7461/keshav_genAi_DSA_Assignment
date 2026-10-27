"""Streamlit front end for the VAE latent-space visualizer.

    streamlit run streamlit_app.py
"""
import numpy as np
import streamlit as st
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.app import LatentSpace, load_or_train_model
from backend.vae import kl_divergence, reconstruction_loss

COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
          '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

st.set_page_config(page_title="VAE Latent Space Visualizer", layout="wide")


@st.cache_resource(show_spinner="Loading the VAE and encoding the test set...")
def load():
    model = load_or_train_model()
    return model, LatentSpace(model)


@st.cache_data(show_spinner=False)
def manifold_images(_space, n=20):
    pts = _space.points
    lo, hi = pts.min(0), pts.max(0)
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(hi[1], lo[1], n)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    z = _space.unproject(grid).astype(np.float32)
    with torch.no_grad():
        imgs = load()[0].decode(torch.from_numpy(z)).numpy().reshape(n, n, 28, 28)
    return imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28), (xs[0], xs[-1], ys[-1], ys[0])


def big(arr, size=224):
    """28x28 float array -> upscaled PIL image (nearest neighbour, so pixels stay crisp)."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return img.resize((size, size), Image.NEAREST)


def image_to_pixels(file):
    """Uploaded picture -> 28x28 float array, light digit on dark background like MNIST."""
    img = Image.open(file).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    scale = 20 / max(img.size)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 255)
    canvas.paste(img, ((28 - w) // 2, (28 - h) // 2))
    px = np.asarray(canvas, dtype=np.float32) / 255.0
    return 1 - px if px.mean() > 0.5 else px  # dark-on-light -> light-on-dark


def run_vae(model, space, pixels, sample):
    x = torch.from_numpy(pixels.astype(np.float32)).view(1, 784)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = model.sample(mu, logvar) if sample else mu
        logits = model.decoder(z)
        recon = torch.sigmoid(logits)[0].numpy().reshape(28, 28)
        rec = reconstruction_loss(logits, x).item()
        kl = kl_divergence(mu, logvar).item()
    return {
        "mu": mu[0].numpy(), "sigma": torch.exp(0.5 * logvar)[0].numpy(), "z": z[0].numpy(),
        "point": space.project(mu.numpy())[0], "recon": recon, "rec_loss": rec, "kl": kl,
    }


def latent_plot(space, point, decoded_point, show_manifold):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    if show_manifold:
        img, extent = manifold_images(space)
        ax.imshow(img, cmap="gray", extent=extent, aspect="auto", alpha=0.45)
    labels = space.labels[space.idx]
    for d in range(10):
        m = labels == d
        ax.scatter(space.points[m, 0], space.points[m, 1], s=7, c=COLORS[d], label=str(d), alpha=0.75)
    ax.scatter(*point, s=260, facecolors="none", edgecolors="black", linewidths=2.2, label="your input", zorder=5)
    if decoded_point is not None:
        ax.scatter(*decoded_point, s=130, marker="D", c="black", edgecolors="white", linewidths=1.5,
                   label="decoded point", zorder=6)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=2, ncol=2, fontsize=8, loc="best", title="digit")
    ax.set_xlim(space.points[:, 0].min() - 0.3, space.points[:, 0].max() + 0.3)
    ax.set_ylim(space.points[:, 1].min() - 0.3, space.points[:, 1].max() + 0.3)
    fig.tight_layout()
    return fig


model, space = load()

st.title("VAE Latent Space Visualizer")
n_params = sum(p.numel() for p in model.parameters())
st.caption(f"Encoder 784 → 512 → 256 → (μ, log σ²) · {model.latent_dim}-D latent · "
           f"decoder back to 784 pixels · {n_params:,} parameters")

# ---------------------------------------------------------------- 1. input
with st.sidebar:
    st.header("1 · Input image")
    source = st.radio("Source", ["MNIST test digit", "Upload an image"])
    pixels, caption = None, ""
    if source == "MNIST test digit":
        digit = st.selectbox("Digit", list(range(10)), index=7)
        rng = np.random.default_rng()
        if "pick" not in st.session_state or st.session_state.get("pick_digit") != digit:
            st.session_state.pick = int(rng.choice(np.where(space.labels == digit)[0]))
            st.session_state.pick_digit = digit
        if st.button("Another random example"):
            st.session_state.pick = int(rng.choice(np.where(space.labels == digit)[0]))
        i = st.session_state.pick
        pixels, caption = space.images[i], f"MNIST test image #{i} (digit {space.labels[i]})"
    else:
        up = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
        st.caption("Dark digit on light paper is inverted automatically.")
        if up is not None:
            try:
                pixels, caption = image_to_pixels(up), "Your uploaded image"
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that image: {e}")
    sample = st.checkbox("Decode a random sample z ~ N(μ, σ²) instead of the mean")

if pixels is None:
    st.info("Choose a digit or upload an image in the sidebar.")
    st.stop()

res = run_vae(model, space, pixels, sample)

# ---------------------------------------------------------------- 2 & 3
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("2 · Original vs reconstruction")
    c1, c2 = st.columns(2)
    c1.image(big(pixels), caption=caption, use_container_width=True)
    c2.image(big(res["recon"]), caption="Reconstruction", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Reconstruction loss (BCE)", f"{res['rec_loss']:.1f}")
    m2.metric("KL divergence", f"{res['kl']:.2f}")
    m3.metric("Total (neg. ELBO)", f"{res['rec_loss'] + res['kl']:.1f}")
    fmt = lambda a: "[" + ", ".join(f"{v:.2f}" for v in a) + "]"
    st.markdown(f"**μ** = `{fmt(res['mu'])}`  \n**σ** = `{fmt(res['sigma'])}`  \n**z** = `{fmt(res['z'])}`")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **Reconstruction loss**: how different the rebuilt image is from the original (lower is better).\n"
            "- **KL divergence**: how far the encoder's distribution N(μ, σ²) is from the standard normal prior "
            "N(0, I). It keeps the latent space smooth and well organised.\n"
            "- **Total**: reconstruction loss + KL, the quantity the VAE minimises during training.")

with right:
    st.subheader("3 · Latent space")
    pts = space.points
    st.caption("Decode any point of the latent plane with the sliders; it appears as a diamond on the plot.")
    s1, s2 = st.columns(2)
    zx = s1.slider("z₁", float(pts[:, 0].min()), float(pts[:, 0].max()), float(res["point"][0]), 0.05)
    zy = s2.slider("z₂", float(pts[:, 1].min()), float(pts[:, 1].max()), float(res["point"][1]), 0.05)
    show_manifold = st.checkbox("Show decoded manifold behind the points")
    z_full = space.unproject(np.array([[zx, zy]], dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        generated = model.decode(torch.from_numpy(z_full))[0].numpy().reshape(28, 28)

    fig = latent_plot(space, res["point"], (zx, zy), show_manifold)
    st.pyplot(fig)
    plt.close(fig)
    g1, g2 = st.columns([1, 3])
    g1.image(big(generated, 112))
    g2.caption(f"Digit generated by the decoder alone from the point ({zx:.2f}, {zy:.2f}).")
    if model.latent_dim != 2:
        st.caption(f"The latent space has {model.latent_dim} dimensions; the plot shows its first two principal components.")
