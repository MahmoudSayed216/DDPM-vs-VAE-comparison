"""
DDPM vs. VAE: CIFAR-10 model comparison.

Loads a trained conditional DDPM checkpoint and a trained conditional VAE checkpoint,
and compares them on:

1. Overall FID / Inception Score (mixed classes, N generated samples per model)
2. Per-class FID / Inception Score, plotted side by side
3. A qualitative check: 10 random DDPM-generated images vs. their 5 nearest real
   CIFAR-10 images (pixel-space L2 distance), to eyeball whether the model is
   generating something new or effectively memorizing training images

Both models are rebuilt using the architecture config saved *inside their own
checkpoint* (checkpoint["config"]), not whatever happens to be in configs.yml on
disk right now -- so this works regardless of what configs.yml has been edited
to since training, and can't hit a shape-mismatch error from an architecture/
checkpoint mismatch.

All graphs are saved as PNG files under OUTPUT_DIR (see CONFIG below) so they're
visible whether this runs in an interactive session or headless (e.g. a Kaggle
script run, a plain `python compare_models.py`).
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
import matplotlib.pyplot as plt

from cifar10_dataset import CIFAR10Dataset, denormalize
from ConditionalDDPM import ConditionalDDPM
from diffusion_utils import GaussianDiffusion
from ConditionalVAE import ConditionalVAE
from metrics import compute_fid_and_is

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ============================================================
# CONFIG -- edit these to match your setup
# ============================================================
DDPM_CHECKPOINT_PATH = "/kaggle/input/models/marwansayed2000/ddpm70i/pytorch/default/1/ddpm.pt"
VAE_CHECKPOINT_PATH = "/kaggle/input/models/marwansayed2000/vae70i/pytorch/default/1/vae.pt"
DATA_DIR = "/kaggle/input/datasets/pankrzysiu/cifar10-python"
DOWNLOAD_DATA = False  # True if DATA_DIR doesn't already contain cifar-10-batches-py

OUTPUT_DIR = "./comparison_outputs"  # where PNG graphs get saved

# ---- Evaluation settings ----
# FID is a biased estimator whose bias shrinks as sample count grows -- estimates
# from only a couple hundred samples are noisy and not directly comparable to
# numbers reported in papers (which typically use 10k-50k samples). These default
# to modest values so the script runs in a reasonable time; raise them for more
# trustworthy numbers if you can afford the extra generation time (DDPM sampling
# in particular is slow -- it runs the full reverse diffusion loop per batch).
N_SAMPLES_OVERALL = 2000      # total generated samples per model for the overall FID/IS estimate
N_SAMPLES_PER_CLASS = 200     # generated samples per class for the per-class FID/IS breakdown
FID_IS_BATCH_SIZE = 100       # generation batch size used internally by compute_fid_and_is

# ---- Nearest-neighbor settings ----
NUM_GENERATED_FOR_NN = 10     # how many DDPM images to generate for the similarity search
NUM_NEIGHBORS = 5             # how many nearest real images to retrieve per generated image

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data loading
# ============================================================
def indices_by_class(dataset, num_classes):
    """torchvision CIFAR10 stores plain python-int labels in .targets."""
    targets = dataset.base_dataset.targets
    buckets = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        buckets[label].append(idx)
    return buckets


def load_data():
    train_dataset = CIFAR10Dataset(root=DATA_DIR, train=True, augment=False, download=DOWNLOAD_DATA)
    test_dataset = CIFAR10Dataset(root=DATA_DIR, train=False, augment=False, download=DOWNLOAD_DATA)

    num_classes = 10
    train_indices_by_class = indices_by_class(train_dataset, num_classes)
    test_indices_by_class = indices_by_class(test_dataset, num_classes)

    print(f"Train set: {len(train_dataset)} images | Test set: {len(test_dataset)} images")
    for c in range(num_classes):
        print(
            f"  class {c} ({CIFAR10_CLASS_NAMES[c]}): "
            f"{len(train_indices_by_class[c])} train / {len(test_indices_by_class[c])} test"
        )

    return train_dataset, test_dataset, train_indices_by_class, test_indices_by_class, num_classes


# ============================================================
# Model loading
# ============================================================
def load_ddpm():
    checkpoint = torch.load(DDPM_CHECKPOINT_PATH, map_location=DEVICE)
    cfg = checkpoint["config"]
    model_cfg = cfg["MODEL"]

    model = ConditionalDDPM(
        num_classes=model_cfg["NUM_CLASSES"],
        embedding_dim=model_cfg["CLASS_EMBEDDING_DIM"],
        num_groups=model_cfg["NUM_GROUPS"],
        channels_per_level=model_cfg["CHANNELS_PER_LEVEL"],
        theta=model_cfg["THETA"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg["DIFFUSION"]["TIMESTEPS"],
        beta_start=cfg["DIFFUSION"]["BETA_START"],
        beta_end=cfg["DIFFUSION"]["BETA_END"],
        schedule=cfg["DIFFUSION"]["BETA_SCHEDULE"],
        device=DEVICE,
    )

    image_size = model_cfg["IMAGE_SIDE_LENGTH"]
    print(f"Loaded DDPM checkpoint from epoch {checkpoint['epoch']} | image size {image_size}x{image_size}")
    return model, diffusion, image_size


def load_vae(expected_image_size):
    checkpoint = torch.load(VAE_CHECKPOINT_PATH, map_location=DEVICE)
    cfg = checkpoint["config"]
    model_cfg = cfg["MODEL"]

    model = ConditionalVAE(
        num_classes=model_cfg["NUM_CLASSES"],
        embedding_dim=model_cfg["CLASS_EMBEDDING_DIM"],
        num_groups=model_cfg["NUM_GROUPS"],
        channels_per_level=model_cfg["CHANNELS_PER_LEVEL"],
        latent_channels=model_cfg["LATENT_CHANNELS"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    latent_channels = model_cfg["LATENT_CHANNELS"]
    latent_spatial = model_cfg["IMAGE_SIDE_LENGTH"] // 8

    print(
        f"Loaded VAE checkpoint from epoch {checkpoint['epoch']} | "
        f"latent shape ({latent_channels}, {latent_spatial}, {latent_spatial})"
    )

    if model_cfg["IMAGE_SIDE_LENGTH"] != expected_image_size:
        print(
            f"WARNING: DDPM image size ({expected_image_size}) and VAE image size "
            f"({model_cfg['IMAGE_SIDE_LENGTH']}) differ -- comparisons below assume matching sizes."
        )

    return model, latent_channels, latent_spatial


# ============================================================
# Sampling functions
#
# Both wrapped to the same interface -- (batch_size, class_ids=None) -> (B, 3, H, W)
# tensor in [-1, 1] -- so the FID/IS code doesn't need to know which model it's
# talking to.
# ============================================================
def make_ddpm_sample_fn(model, diffusion, image_size, num_classes):
    @torch.no_grad()
    def ddpm_sample(batch_size, class_ids=None):
        if class_ids is None:
            class_ids = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
        elif not torch.is_tensor(class_ids):
            class_ids = torch.as_tensor(class_ids, device=DEVICE, dtype=torch.long)
        samples = diffusion.p_sample_loop(model, (batch_size, 3, image_size, image_size), class_ids, DEVICE)
        return samples.clamp(-1.0, 1.0)

    return ddpm_sample


def make_vae_sample_fn(model, latent_channels, latent_spatial, num_classes):
    @torch.no_grad()
    def vae_sample(batch_size, class_ids=None):
        if class_ids is None:
            class_ids = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
        elif not torch.is_tensor(class_ids):
            class_ids = torch.as_tensor(class_ids, device=DEVICE, dtype=torch.long)
        z = torch.randn(batch_size, latent_channels, latent_spatial, latent_spatial, device=DEVICE)
        samples = model.decode(z, class_ids)
        return samples.clamp(-1.0, 1.0)

    return vae_sample


def make_class_conditional_sample_fn(sample_fn, class_id):
    """Wraps a (batch_size, class_ids=None)->images sampler into a plain
    (batch_size)->images closure fixed to one class, for compute_fid_and_is."""
    def _fn(batch_size):
        class_ids = torch.full((batch_size,), class_id, device=DEVICE, dtype=torch.long)
        return sample_fn(batch_size, class_ids=class_ids)
    return _fn


# ============================================================
# FID / IS computation
# ============================================================
def compute_overall_metrics(ddpm_sample, vae_sample, test_dataset):
    real_eval_loader = DataLoader(test_dataset, batch_size=FID_IS_BATCH_SIZE, shuffle=True, num_workers=2)

    print("Computing overall FID/IS for DDPM (runs the full reverse diffusion loop -- can be slow)...")
    ddpm_fid, ddpm_is_mean, ddpm_is_std = compute_fid_and_is(
        ddpm_sample, real_eval_loader, num_samples=N_SAMPLES_OVERALL, device=DEVICE, batch_size=FID_IS_BATCH_SIZE,
    )
    print(f"DDPM overall: FID={ddpm_fid:.3f} | IS={ddpm_is_mean:.3f} +/- {ddpm_is_std:.3f}")

    print("Computing overall FID/IS for VAE...")
    vae_fid, vae_is_mean, vae_is_std = compute_fid_and_is(
        vae_sample, real_eval_loader, num_samples=N_SAMPLES_OVERALL, device=DEVICE, batch_size=FID_IS_BATCH_SIZE,
    )
    print(f"VAE overall:  FID={vae_fid:.3f} | IS={vae_is_mean:.3f} +/- {vae_is_std:.3f}")

    return {
        "DDPM": {"fid": ddpm_fid, "is_mean": ddpm_is_mean, "is_std": ddpm_is_std},
        "VAE": {"fid": vae_fid, "is_mean": vae_is_mean, "is_std": vae_is_std},
    }


def compute_per_class_metrics(ddpm_sample, vae_sample, test_dataset, test_indices_by_class, num_classes):
    if N_SAMPLES_PER_CLASS < 200:
        print(
            f"WARNING: N_SAMPLES_PER_CLASS={N_SAMPLES_PER_CLASS} is quite small -- FID estimates "
            f"get noisy/biased below a few hundred samples. Treat per-class numbers as directional, "
            f"not precise, unless you raise this."
        )

    results = []
    for model_name, sample_fn in [("DDPM", ddpm_sample), ("VAE", vae_sample)]:
        for class_id in range(num_classes):
            real_subset = Subset(test_dataset, test_indices_by_class[class_id])
            real_loader = DataLoader(real_subset, batch_size=min(FID_IS_BATCH_SIZE, len(real_subset)), shuffle=True)

            class_sample_fn = make_class_conditional_sample_fn(sample_fn, class_id)
            fid_value, is_mean, is_std = compute_fid_and_is(
                class_sample_fn, real_loader, num_samples=N_SAMPLES_PER_CLASS, device=DEVICE,
                batch_size=min(FID_IS_BATCH_SIZE, N_SAMPLES_PER_CLASS),
            )

            results.append({
                "model": model_name, "class": class_id, "class_name": CIFAR10_CLASS_NAMES[class_id],
                "fid": fid_value, "is_mean": is_mean, "is_std": is_std,
            })
            print(f"[{model_name}] {CIFAR10_CLASS_NAMES[class_id]:>10s}: FID={fid_value:.3f} | IS={is_mean:.3f} +/- {is_std:.3f}")

    return pd.DataFrame(results)


# ============================================================
# Plotting -- every figure is saved to OUTPUT_DIR as a PNG
# ============================================================
def plot_per_class_metrics(per_class_df, num_classes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(num_classes)
    width = 0.35

    ddpm_fid = per_class_df[per_class_df["model"] == "DDPM"].sort_values("class")["fid"].values
    vae_fid = per_class_df[per_class_df["model"] == "VAE"].sort_values("class")["fid"].values

    axes[0].bar(x - width / 2, ddpm_fid, width, label="DDPM")
    axes[0].bar(x + width / 2, vae_fid, width, label="VAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CIFAR10_CLASS_NAMES, rotation=45, ha="right")
    axes[0].set_ylabel("FID (lower is better)")
    axes[0].set_title("Per-class FID")
    axes[0].legend()

    ddpm_is_mean = per_class_df[per_class_df["model"] == "DDPM"].sort_values("class")["is_mean"].values
    ddpm_is_std = per_class_df[per_class_df["model"] == "DDPM"].sort_values("class")["is_std"].values
    vae_is_mean = per_class_df[per_class_df["model"] == "VAE"].sort_values("class")["is_mean"].values
    vae_is_std = per_class_df[per_class_df["model"] == "VAE"].sort_values("class")["is_std"].values

    axes[1].bar(x - width / 2, ddpm_is_mean, width, yerr=ddpm_is_std, capsize=3, label="DDPM")
    axes[1].bar(x + width / 2, vae_is_mean, width, yerr=vae_is_std, capsize=3, label="VAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CIFAR10_CLASS_NAMES, rotation=45, ha="right")
    axes[1].set_ylabel("Inception Score (higher is better)")
    axes[1].set_title("Per-class Inception Score")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved -> {save_path}")
    plt.show()
    plt.close(fig)


def plot_overall_metrics(overall_results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ["DDPM", "VAE"]
    fid_values = [overall_results[m]["fid"] for m in models]
    is_means = [overall_results[m]["is_mean"] for m in models]
    is_stds = [overall_results[m]["is_std"] for m in models]

    axes[0].bar(models, fid_values, color=["tab:blue", "tab:orange"])
    axes[0].set_ylabel("FID (lower is better)")
    axes[0].set_title(f"Overall FID (N={N_SAMPLES_OVERALL})")

    axes[1].bar(models, is_means, yerr=is_stds, capsize=5, color=["tab:blue", "tab:orange"])
    axes[1].set_ylabel("Inception Score (higher is better)")
    axes[1].set_title(f"Overall Inception Score (N={N_SAMPLES_OVERALL})")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "overall_metrics.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved -> {save_path}")
    plt.show()
    plt.close(fig)

    print(pd.DataFrame(overall_results).T)


# ============================================================
# Nearest-neighbor check: DDPM samples vs. real CIFAR-10
#
# Searches the full CIFAR-10 *training* set (what the model actually learned
# from) using simple pixel-space L2 distance. This is a coarse similarity
# metric -- it won't catch semantic similarity the way a learned feature space
# would -- but it directly answers the practical question: is the model
# reproducing something close to a specific training image, or generating
# something genuinely new?
# ============================================================
def run_nearest_neighbor_check(ddpm_sample, train_dataset, num_classes):
    gen_class_ids = torch.randint(0, num_classes, (NUM_GENERATED_FOR_NN,), device=DEVICE)
    generated_images = ddpm_sample(NUM_GENERATED_FOR_NN, class_ids=gen_class_ids)
    generated_images_01 = denormalize(generated_images).cpu()

    print(
        f"Generated {NUM_GENERATED_FOR_NN} DDPM images for classes: "
        f"{[CIFAR10_CLASS_NAMES[c] for c in gen_class_ids.cpu().tolist()]}"
    )

    real_bank_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=2)

    real_images_01_chunks = []
    real_labels_chunks = []
    for imgs, labels in real_bank_loader:
        real_images_01_chunks.append(denormalize(imgs))
        real_labels_chunks.append(labels)

    real_images_01 = torch.cat(real_images_01_chunks, dim=0)
    real_labels = torch.cat(real_labels_chunks, dim=0)
    print(f"Real image search bank: {real_images_01.shape[0]} training images")

    gen_flat = generated_images_01.view(NUM_GENERATED_FOR_NN, -1)
    real_flat = real_images_01.view(real_images_01.shape[0], -1)

    # Chunked distance computation so peak memory stays bounded regardless of bank size.
    chunk_size = 5000
    all_distances = torch.empty(NUM_GENERATED_FOR_NN, real_flat.shape[0])
    for start in range(0, real_flat.shape[0], chunk_size):
        end = min(start + chunk_size, real_flat.shape[0])
        all_distances[:, start:end] = torch.cdist(gen_flat, real_flat[start:end])

    neighbor_distances, neighbor_indices = torch.topk(all_distances, k=NUM_NEIGHBORS, largest=False, dim=1)

    fig, axes = plt.subplots(
        NUM_GENERATED_FOR_NN, NUM_NEIGHBORS + 1,
        figsize=(2.2 * (NUM_NEIGHBORS + 1), 2.2 * NUM_GENERATED_FOR_NN),
    )

    for row in range(NUM_GENERATED_FOR_NN):
        ax = axes[row, 0]
        ax.imshow(generated_images_01[row].permute(1, 2, 0).numpy())
        ax.set_title(f"Generated\n({CIFAR10_CLASS_NAMES[gen_class_ids[row].item()]})", fontsize=9)
        ax.axis("off")

        for col in range(NUM_NEIGHBORS):
            real_idx = neighbor_indices[row, col].item()
            dist = neighbor_distances[row, col].item()
            real_label = CIFAR10_CLASS_NAMES[real_labels[real_idx].item()]

            ax = axes[row, col + 1]
            ax.imshow(real_images_01[real_idx].permute(1, 2, 0).numpy())
            ax.set_title(f"#{col + 1}: {real_label}\nd={dist:.2f}", fontsize=8)
            ax.axis("off")

    plt.suptitle("DDPM-generated (left) vs. 5 nearest real CIFAR-10 images (pixel-space L2)", y=1.001)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "nearest_neighbors.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Orchestration
# ============================================================
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use a non-interactive backend if there's no display attached, so plt.show()
    # never blocks or errors out in a headless run -- the PNGs saved above are
    # what actually guarantee the graphs are visible either way.
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")

    train_dataset, test_dataset, train_indices_by_class, test_indices_by_class, num_classes = load_data()

    ddpm_model, diffusion, image_size = load_ddpm()
    vae_model, vae_latent_channels, vae_latent_spatial = load_vae(expected_image_size=image_size)

    ddpm_sample = make_ddpm_sample_fn(ddpm_model, diffusion, image_size, num_classes)
    vae_sample = make_vae_sample_fn(vae_model, vae_latent_channels, vae_latent_spatial, num_classes)

    overall_results = compute_overall_metrics(ddpm_sample, vae_sample, test_dataset)
    per_class_df = compute_per_class_metrics(ddpm_sample, vae_sample, test_dataset, test_indices_by_class, num_classes)

    plot_per_class_metrics(per_class_df, num_classes)
    plot_overall_metrics(overall_results)

    run_nearest_neighbor_check(ddpm_sample, train_dataset, num_classes)

    print("\n--- Caveats ---")
    print("- FID/IS at small N: raise N_SAMPLES_OVERALL / N_SAMPLES_PER_CLASS for numbers you'd")
    print("  trust in a report; the defaults here favor a fast run over precision.")
    print("- Per-class FID reference size: the CIFAR-10 test split only has 1000 images per class,")
    print("  a smaller, noisier real-image reference than the overall FID (full 10k test set).")
    print("- DDPM sampling is slow: every generated image runs a full reverse diffusion loop.")
    print("- Nearest-neighbor distance is pixel-space L2, not a learned similarity metric -- a")
    print("  blunt but interpretable memorization check, not proof either way.")


if __name__ == "__main__":
    main()