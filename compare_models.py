"""
compare_models.py
==================

Compares the trained Conditional DDPM and Conditional VAE on CIFAR-10:

  1. Loads both model checkpoints.
  2. Generates N samples from each model, N/10 per class.
  3. Computes overall FID + Inception Score, and per-class FID + IS, for both models.
  4. Plots the comparisons (saved as PNGs, and shown inline if run in a notebook cell).
  5. For the DDPM, takes one generated image per class (10 total) and finds the
     top-5 most similar real CIFAR-10 images (Inception-feature cosine similarity).
  6. Prints a brief text conclusion comparing the two models.

Assumed repo layout (override any of these with CLI flags if yours differs):

    <repo_root>/
        ddpm/
            cifar10_dataset.py
            ConditionalDDPM.py
            diffusion_utils.py
            configs.yml
            checkpoints/ddpm_epoch_XXX.pt
        vae/
            cifar10_dataset.py
            ConditionalVAE.py
            configs.yml
            checkpoints/vae_epoch_XXX.pt
        compare_models.py   <- this file

Usage (from the repo root, e.g. in a Kaggle notebook cell):

    !python compare_models.py

Or, to see the figures rendered inline in the notebook (instead of only saved
to disk), run it as a magic command instead of a shell call:

    %run compare_models.py

All settings live in the CONFIG block right below the imports -- edit those
values directly instead of passing CLI flags.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = 10


# ====================================================================
# CONFIG -- edit these directly instead of passing command-line flags.
# ====================================================================

# Directories containing each project's files (dataset/model code, configs.yml,
# checkpoints/). Assumed layout: <repo_root>/ddpm/ and <repo_root>/vae/, with
# this script sitting at <repo_root>/compare_models.py.
DDPM_DIR = "./"
VAE_DIR = "./"

# Path to each configs.yml. Set to None to default to "<DDPM_DIR>/configs.yml"
# and "<VAE_DIR>/configs.yml".
DDPM_CONFIG_PATH = "ddpm_configs.yml"
VAE_CONFIG_PATH = "vae_configs.yml"

# Path to a specific checkpoint file. Set to None to auto-pick the
# highest-epoch "ddpm_epoch_*.pt" / "vae_epoch_*.pt" checkpoint found in each
# model's configured CHECKPOINT_DIR.
DDPM_CKPT_PATH = "/kaggle/input/models/marwansayed2000/ddpm70i/pytorch/default/1/ddpm.pt"
VAE_CKPT_PATH = "/kaggle/input/models/marwansayed2000/vae70i/pytorch/default/1/vae.pt"

# Root dir for the real CIFAR-10 data used as the FID/IS reference and the
# nearest-neighbor pool. Set to None to default to DATA.DATA_DIR from
# DDPM_CONFIG_PATH.
REAL_DATA_DIR = "/kaggle/input/datasets/pankrzysiu/cifar10-python"

# Whether to allow downloading CIFAR-10 if it isn't found locally (leave
# False on a read-only Kaggle input mount).
DOWNLOAD = False

# Total number of samples to generate PER MODEL (rounded down to a multiple
# of 10 so every class gets an equal share). Raise this for tighter FID/IS
# estimates if you have the compute budget.
N_SAMPLES = 5000

# Batch size used during sample generation (mainly matters for the DDPM,
# since each batch runs the full reverse-diffusion loop).
GEN_BATCH_SIZE = 64

# Cap on real test images used per class as the FID/IS reference (the test
# split only has ~1000/class, so this rarely needs to change).
MAX_REAL_PER_CLASS = 1000

# How many real train-split images per class to search over for the
# nearest-neighbor step.
NN_POOL_PER_CLASS = 1000

# Where to save the output PNG figures.
OUTPUT_DIR = "comparison_outputs"

# Random seed, for reproducible sample generation.
SEED = 46

# "cuda", "cpu", or None to auto-detect (cuda if available, else cpu).
DEVICE = "cuda"


# ------------------------------------------------------------------
# Dynamic module loading (so we can import ddpm/ and vae/ code, which
# both define modules with the same names, without them clobbering
# each other in sys.modules).
# ------------------------------------------------------------------

def _load_module(unique_name, file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Expected to find '{file_path}' but it doesn't exist. "
            f"Pass the correct directory via the CLI flags (see --help)."
        )
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_relative(base_dir, maybe_relative_path):
    """configs.yml paths (e.g. CHECKPOINT_DIR: './checkpoints') are relative to
    wherever the original training script was run from. We treat them as
    relative to the model's own directory (base_dir) unless already absolute."""
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    return os.path.normpath(os.path.join(base_dir, maybe_relative_path))


def _find_latest_checkpoint(checkpoint_dir, prefix):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def epoch_num(fname):
        try:
            return int(fname[len(prefix):-len(".pt")])
        except ValueError:
            return -1

    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching '{prefix}*.pt' found in {checkpoint_dir}")
    ckpts.sort(key=epoch_num)
    return os.path.join(checkpoint_dir, ckpts[-1])


# ------------------------------------------------------------------
# FID / IS helpers
# ------------------------------------------------------------------

def _to_uint8(images):
    images = (images.clamp(-1, 1) + 1.0) / 2.0
    return (images * 255).to(torch.uint8)


@torch.no_grad()
def compute_fid_and_is(real_imgs, fake_imgs, device, batch_size=200):
    """real_imgs, fake_imgs: (N, 3, H, W) tensors in [-1, 1] on CPU."""
    fid = FrechetInceptionDistance(normalize=False).to(device)
    inc = InceptionScore(normalize=False).to(device)

    for i in range(0, real_imgs.shape[0], batch_size):
        batch = real_imgs[i:i + batch_size].to(device)
        fid.update(_to_uint8(batch), real=True)

    for i in range(0, fake_imgs.shape[0], batch_size):
        batch = fake_imgs[i:i + batch_size].to(device)
        u8 = _to_uint8(batch)
        fid.update(u8, real=False)
        inc.update(u8)

    fid_value = fid.compute().item()
    is_mean, is_std = inc.compute()
    del fid, inc
    return fid_value, is_mean.item(), is_std.item()


# ------------------------------------------------------------------
# Inception-feature embeddings, used only for the nearest-neighbor
# similarity lookup (kept separate from the FID/IS metrics above).
# ------------------------------------------------------------------

class InceptionEmbedder:
    def __init__(self, device):
        from torchvision.models import inception_v3
        try:
            from torchvision.models import Inception_V3_Weights
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        except ImportError:
            net = inception_v3(pretrained=True, aux_logits=True)
        net.fc = torch.nn.Identity()
        net.eval().to(device)
        self.net = net
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, images_neg1_to_1, batch_size=128):
        """images_neg1_to_1: (N, 3, 32, 32) tensor in [-1, 1] on CPU. Returns (N, 2048) CPU tensor."""
        feats = []
        for i in range(0, images_neg1_to_1.shape[0], batch_size):
            batch = images_neg1_to_1[i:i + batch_size].to(self.device)
            batch = (batch.clamp(-1, 1) + 1.0) / 2.0  # -> [0, 1]
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - self.mean) / self.std
            out = self.net(batch)
            feats.append(out.cpu())
        return torch.cat(feats, dim=0)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_overall_metrics(ddpm_fid, ddpm_is, vae_fid, vae_is, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    models = ["DDPM", "VAE"]
    fids = [ddpm_fid, vae_fid]
    axes[0].bar(models, fids, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Overall FID (lower is better)")
    axes[0].set_ylabel("FID")
    for i, v in enumerate(fids):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    is_means = [ddpm_is[0], vae_is[0]]
    is_stds = [ddpm_is[1], vae_is[1]]
    axes[1].bar(models, is_means, yerr=is_stds, capsize=6, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Overall Inception Score (higher is better)")
    axes[1].set_ylabel("IS")
    for i, v in enumerate(is_means):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    path = os.path.join(output_dir, "overall_metrics.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_per_class_metric(ddpm_values, vae_values, ylabel, title, filename, output_dir, yerr_ddpm=None, yerr_vae=None):
    x = np.arange(NUM_CLASSES)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, ddpm_values, width, yerr=yerr_ddpm, capsize=4, label="DDPM", color="#4C72B0")
    ax.bar(x + width / 2, vae_values, width, yerr=yerr_vae, capsize=4, label="VAE", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_nn_grid(gen_images, neighbor_images, neighbor_sims, output_dir):
    """gen_images: list of 10 (3,32,32) tensors in [-1,1].
    neighbor_images: list of 10 tensors (5,3,32,32) in [-1,1].
    neighbor_sims: list of 10 arrays of length 5 (cosine similarities)."""
    n_cols = 6  # generated + top-5
    fig, axes = plt.subplots(NUM_CLASSES, n_cols, figsize=(n_cols * 1.6, NUM_CLASSES * 1.7))

    def to_disp(img):
        img = (img.clamp(-1, 1) + 1.0) / 2.0
        return img.permute(1, 2, 0).numpy()

    for row in range(NUM_CLASSES):
        axes[row, 0].imshow(to_disp(gen_images[row]))
        axes[row, 0].set_ylabel(CIFAR10_CLASS_NAMES[row], fontsize=9)
        if row == 0:
            axes[row, 0].set_title("DDPM sample", fontsize=9)
        for col in range(5):
            ax = axes[row, col + 1]
            ax.imshow(to_disp(neighbor_images[row][col]))
            ax.set_title(f"sim={neighbor_sims[row][col]:.2f}", fontsize=8)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("DDPM sample vs. top-5 nearest real CIFAR-10 images (Inception-feature cosine similarity)")
    fig.tight_layout()
    path = os.path.join(output_dir, "ddpm_nearest_neighbors.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved -> {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ddpm_dir = os.path.abspath(DDPM_DIR)
    vae_dir = os.path.abspath(VAE_DIR)
    ddpm_config_path = DDPM_CONFIG_PATH or os.path.join(ddpm_dir, "configs.yml")
    vae_config_path = VAE_CONFIG_PATH or os.path.join(vae_dir, "configs.yml")

    with open(ddpm_config_path, "r") as f:
        ddpm_cfg = yaml.safe_load(f)
    with open(vae_config_path, "r") as f:
        vae_cfg = yaml.safe_load(f)

    # ---- Dynamically import each project's modules ----
    ddpm_model_mod = _load_module("ddpm_ConditionalDDPM", os.path.join(ddpm_dir, "ConditionalDDPM.py"))
    diffusion_mod = _load_module("ddpm_diffusion_utils", os.path.join(ddpm_dir, "diffusion_utils.py"))
    vae_model_mod = _load_module("vae_ConditionalVAE", os.path.join(vae_dir, "ConditionalVAE.py"))
    dataset_mod = _load_module("shared_cifar10_dataset", os.path.join(ddpm_dir, "cifar10_dataset.py"))

    CIFAR10Dataset = dataset_mod.CIFAR10Dataset

    # ---- Build + load DDPM ----
    dm_cfg = ddpm_cfg["MODEL"]
    ddpm_model = ddpm_model_mod.ConditionalDDPM(
        num_classes=dm_cfg["NUM_CLASSES"],
        embedding_dim=dm_cfg["CLASS_EMBEDDING_DIM"],
        num_groups=dm_cfg["NUM_GROUPS"],
        channels_per_level=dm_cfg["CHANNELS_PER_LEVEL"],
        theta=dm_cfg["THETA"],
    ).to(device)

    ddpm_ckpt_dir = _resolve_relative(ddpm_dir, ddpm_cfg["TRAINING"]["CHECKPOINT_DIR"])
    ddpm_ckpt_path = DDPM_CKPT_PATH or _find_latest_checkpoint(ddpm_ckpt_dir, "ddpm_epoch_")
    print(f"Loading DDPM checkpoint: {ddpm_ckpt_path}")
    ddpm_ckpt = torch.load(ddpm_ckpt_path, map_location=device)
    ddpm_model.load_state_dict(ddpm_ckpt["model_state_dict"])
    ddpm_model.eval()

    diffusion = diffusion_mod.GaussianDiffusion(
        timesteps=ddpm_cfg["DIFFUSION"]["TIMESTEPS"],
        beta_start=ddpm_cfg["DIFFUSION"]["BETA_START"],
        beta_end=ddpm_cfg["DIFFUSION"]["BETA_END"],
        schedule=ddpm_cfg["DIFFUSION"]["BETA_SCHEDULE"],
        device=device,
    )

    # ---- Build + load VAE ----
    vm_cfg = vae_cfg["MODEL"]
    vae_model = vae_model_mod.ConditionalVAE(
        num_classes=vm_cfg["NUM_CLASSES"],
        embedding_dim=vm_cfg["CLASS_EMBEDDING_DIM"],
        num_groups=vm_cfg["NUM_GROUPS"],
        channels_per_level=vm_cfg["CHANNELS_PER_LEVEL"],
        latent_channels=vm_cfg["LATENT_CHANNELS"],
    ).to(device)

    vae_ckpt_dir = _resolve_relative(vae_dir, vae_cfg["TRAINING"]["CHECKPOINT_DIR"])
    vae_ckpt_path = VAE_CKPT_PATH or _find_latest_checkpoint(vae_ckpt_dir, "vae_epoch_")
    print(f"Loading VAE checkpoint: {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    vae_latent_channels = vm_cfg["LATENT_CHANNELS"]
    vae_latent_spatial = vm_cfg["IMAGE_SIDE_LENGTH"] // 8

    # ---- Sample counts ----
    n_total = (N_SAMPLES // NUM_CLASSES) * NUM_CLASSES
    per_class = n_total // NUM_CLASSES
    if n_total != N_SAMPLES:
        print(f"Rounding N_SAMPLES {N_SAMPLES} down to {n_total} ({per_class}/class).")

    # ---- Generation helpers ----
    image_size = dm_cfg["IMAGE_SIDE_LENGTH"]

    @torch.no_grad()
    def ddpm_generate(class_id, n):
        out = []
        remaining = n
        while remaining > 0:
            b = min(GEN_BATCH_SIZE, remaining)
            class_ids = torch.full((b,), class_id, dtype=torch.long, device=device)
            samples = diffusion.p_sample_loop(ddpm_model, (b, 3, image_size, image_size), class_ids, device)
            out.append(samples.cpu())
            remaining -= b
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def vae_generate(class_id, n):
        out = []
        remaining = n
        vae_model.eval()
        while remaining > 0:
            b = min(max(GEN_BATCH_SIZE, 200), remaining)
            z = torch.randn(b, vae_latent_channels, vae_latent_spatial, vae_latent_spatial, device=device)
            class_ids = torch.full((b,), class_id, dtype=torch.long, device=device)
            samples = vae_model.decode(z, class_ids).clamp(-1.0, 1.0)
            out.append(samples.cpu())
            remaining -= b
        return torch.cat(out, dim=0)

    # ---- Generate per-class samples for both models ----
    print(f"Generating {per_class} samples/class ({n_total} total) from each model...")
    ddpm_samples_by_class = {}
    vae_samples_by_class = {}
    for c in range(NUM_CLASSES):
        t0 = time.time()
        ddpm_samples_by_class[c] = ddpm_generate(c, per_class)
        vae_samples_by_class[c] = vae_generate(c, per_class)
        print(f"  class {c} ({CIFAR10_CLASS_NAMES[c]}): done in {time.time() - t0:.1f}s")

    ddpm_all = torch.cat([ddpm_samples_by_class[c] for c in range(NUM_CLASSES)], dim=0)
    vae_all = torch.cat([vae_samples_by_class[c] for c in range(NUM_CLASSES)], dim=0)

    # ---- Real reference images (test split), bucketed by class ----
    real_data_dir = REAL_DATA_DIR or ddpm_cfg["DATA"]["DATA_DIR"]
    real_test_dataset = CIFAR10Dataset(
        root=real_data_dir, train=False, image_side_length=image_size, augment=False, download=DOWNLOAD,
    )
    real_loader = DataLoader(real_test_dataset, batch_size=500, shuffle=False, num_workers=2)

    real_by_class = {c: [] for c in range(NUM_CLASSES)}
    for imgs, labels in real_loader:
        for img, lbl in zip(imgs, labels):
            c = int(lbl.item())
            if len(real_by_class[c]) < MAX_REAL_PER_CLASS:
                real_by_class[c].append(img)
    real_by_class = {c: torch.stack(v, dim=0) for c, v in real_by_class.items()}
    real_all = torch.cat([real_by_class[c] for c in range(NUM_CLASSES)], dim=0)
    print(f"Loaded {real_all.shape[0]} real test images ({[real_by_class[c].shape[0] for c in range(NUM_CLASSES)]} per class).")

    # ---- Overall FID / IS ----
    print("Computing overall FID/IS for DDPM...")
    ddpm_fid, ddpm_is_mean, ddpm_is_std = compute_fid_and_is(real_all, ddpm_all, device)
    print(f"  DDPM overall: FID={ddpm_fid:.3f} | IS={ddpm_is_mean:.3f} +/- {ddpm_is_std:.3f}")

    print("Computing overall FID/IS for VAE...")
    vae_fid, vae_is_mean, vae_is_std = compute_fid_and_is(real_all, vae_all, device)
    print(f"  VAE overall: FID={vae_fid:.3f} | IS={vae_is_mean:.3f} +/- {vae_is_std:.3f}")

    # ---- Per-class FID / IS ----
    ddpm_fid_pc, ddpm_is_mean_pc, ddpm_is_std_pc = [], [], []
    vae_fid_pc, vae_is_mean_pc, vae_is_std_pc = [], [], []
    print("Computing per-class FID/IS (this repeats the metric computation 10x per model)...")
    for c in range(NUM_CLASSES):
        f, m, s = compute_fid_and_is(real_by_class[c], ddpm_samples_by_class[c], device)
        ddpm_fid_pc.append(f); ddpm_is_mean_pc.append(m); ddpm_is_std_pc.append(s)

        f, m, s = compute_fid_and_is(real_by_class[c], vae_samples_by_class[c], device)
        vae_fid_pc.append(f); vae_is_mean_pc.append(m); vae_is_std_pc.append(s)
        print(f"  class {c} ({CIFAR10_CLASS_NAMES[c]}): "
              f"DDPM FID={ddpm_fid_pc[-1]:.2f} IS={ddpm_is_mean_pc[-1]:.2f} | "
              f"VAE FID={vae_fid_pc[-1]:.2f} IS={vae_is_mean_pc[-1]:.2f}")

    # ---- Plots ----
    plot_overall_metrics(ddpm_fid, (ddpm_is_mean, ddpm_is_std), vae_fid, (vae_is_mean, vae_is_std), OUTPUT_DIR)
    plot_per_class_metric(ddpm_fid_pc, vae_fid_pc, "FID", "Per-class FID (lower is better)",
                           "per_class_fid.png", OUTPUT_DIR)
    plot_per_class_metric(ddpm_is_mean_pc, vae_is_mean_pc, "Inception Score", "Per-class Inception Score (higher is better)",
                           "per_class_is.png", OUTPUT_DIR,
                           yerr_ddpm=ddpm_is_std_pc, yerr_vae=vae_is_std_pc)

    # ---- DDPM nearest-neighbor lookup ----
    print("Building Inception embedder for nearest-neighbor lookup...")
    embedder = InceptionEmbedder(device)

    real_train_dataset = CIFAR10Dataset(
        root=real_data_dir, train=True, image_side_length=image_size, augment=False, download=DOWNLOAD,
    )
    train_loader = DataLoader(real_train_dataset, batch_size=500, shuffle=False, num_workers=2)
    train_pool_by_class = {c: [] for c in range(NUM_CLASSES)}
    for imgs, labels in train_loader:
        for img, lbl in zip(imgs, labels):
            c = int(lbl.item())
            if len(train_pool_by_class[c]) < NN_POOL_PER_CLASS:
                train_pool_by_class[c].append(img)
        if all(len(v) >= NN_POOL_PER_CLASS for v in train_pool_by_class.values()):
            break
    train_pool_by_class = {c: torch.stack(v, dim=0) for c, v in train_pool_by_class.items()}

    gen_representative = [ddpm_samples_by_class[c][0] for c in range(NUM_CLASSES)]
    gen_embeds = embedder.embed(torch.stack(gen_representative, dim=0))  # (10, 2048)

    nn_images, nn_sims = [], []
    print("Searching for top-5 nearest real images per class...")
    for c in range(NUM_CLASSES):
        pool = train_pool_by_class[c]
        pool_embeds = embedder.embed(pool)  # (pool_size, 2048)

        g = F.normalize(gen_embeds[c:c + 1], dim=1)
        p = F.normalize(pool_embeds, dim=1)
        sims = (g @ p.T).squeeze(0)  # (pool_size,)

        top5 = torch.topk(sims, k=5)
        nn_images.append(pool[top5.indices])
        nn_sims.append(top5.values.numpy())

    plot_nn_grid(gen_representative, nn_images, nn_sims, OUTPUT_DIR)

    # ---- Brief conclusion ----
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    fid_winner = "DDPM" if ddpm_fid < vae_fid else "VAE"
    is_winner = "DDPM" if ddpm_is_mean > vae_is_mean else "VAE"
    worst_ddpm_class = CIFAR10_CLASS_NAMES[int(np.argmax(ddpm_fid_pc))]
    worst_vae_class = CIFAR10_CLASS_NAMES[int(np.argmax(vae_fid_pc))]

    print(
        f"Overall, {fid_winner} produces more realistic samples by FID "
        f"(DDPM={ddpm_fid:.2f} vs VAE={vae_fid:.2f}, lower is better), while "
        f"{is_winner} scores higher on Inception Score "
        f"(DDPM={ddpm_is_mean:.2f}+/-{ddpm_is_std:.2f} vs VAE={vae_is_mean:.2f}+/-{vae_is_std:.2f}, "
        f"higher is better -- more confident/diverse class predictions). "
        f"Per-class FID shows the DDPM struggles most with '{worst_ddpm_class}' and the VAE with "
        f"'{worst_vae_class}', suggesting those categories are harder for each architecture to model. "
        f"Note these scores are computed on only {per_class} generated images per class, so per-class "
        f"FID/IS in particular should be read as a rough, somewhat noisy signal rather than a precise "
        f"estimate -- increase N_SAMPLES at the top of the script for a tighter comparison if you have "
        f"the compute budget."
    )

    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()