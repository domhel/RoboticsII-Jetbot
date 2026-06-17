"""Find good data augmentations to close the camera-domain gap.

We train on ``clean_data`` (recorded with a *different* camera) but deploy on
our own camera, whose sample images live in
``downloaded/dataset_xy_msdmhb``. The two cameras look at the same track but
differ in colour balance / exposure (our camera has a strong magenta cast).

This script:
  1. measures the colour statistics of the *target* domain (our camera) and the
     *source* domain (clean_data),
  2. defines a set of candidate augmentations (brightness, contrast, saturation,
     hue shift, white-balance channel jitter, blur, noise and a combined
     pipeline),
  3. applies each candidate to clean_data samples and scores how well the
     augmented source distribution *covers* the target distribution,
  4. saves visual grids and a ranked summary under ``augmentation_search/``.

It uses only PIL / numpy / matplotlib so it runs in the local uv environment
(no torch / CUDA needed). The winning pipeline is mirrored in
``train_model-2.ipynb`` so the model trained on clean_data generalises to our
camera.
"""

from __future__ import annotations

import glob
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "clean_data")  # training domain (other camera)
TARGET_DIR = os.path.join(HERE, "downloaded", "dataset_xy_msdmhb")  # our camera
OUTPUT_DIR = os.path.join(HERE, "augmentation_search")
SIZE = 224
SEED = 0


# --------------------------------------------------------------------------- #
# Augmentation primitives (PIL based, deterministic given an rng)
# --------------------------------------------------------------------------- #
def _enh(img, factor, kind):
    enhancer = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "saturation": ImageEnhance.Color,
    }[kind]
    return enhancer(img).enhance(factor)


def hue_shift(img: Image.Image, delta: float) -> Image.Image:
    """Rotate hue by ``delta`` in [-0.5, 0.5] (fraction of the colour wheel)."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(delta * 255)) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def white_balance(img: Image.Image, gains: tuple[float, float, float]) -> Image.Image:
    """Multiply each RGB channel by a gain (simulates white-balance drift)."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * np.array(gains, dtype=np.float32).reshape(1, 1, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_noise(img: Image.Image, std: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = arr + np_rng.normal(0, std * 255, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Candidate augmentation pipelines. Each is a callable (img, rng) -> img.
# --------------------------------------------------------------------------- #
def aug_none(img, rng):
    return img


def aug_brightness_contrast(img, rng):
    img = _enh(img, rng.uniform(0.6, 1.4), "brightness")
    img = _enh(img, rng.uniform(0.6, 1.4), "contrast")
    return img


def aug_saturation(img, rng):
    return _enh(img, rng.uniform(0.5, 1.5), "saturation")


def aug_hue(img, rng):
    return hue_shift(img, rng.uniform(-0.15, 0.15))


def aug_white_balance(img, rng):
    gains = (rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2))
    return white_balance(img, gains)


def aug_blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.5)))


def aug_noise(img, rng):
    return add_noise(img, std=rng.uniform(0.0, 0.04), rng=rng)


def aug_combined(img, rng):
    """The recommended pipeline: colour + white-balance + blur + noise."""
    img = _enh(img, rng.uniform(0.6, 1.4), "brightness")
    img = _enh(img, rng.uniform(0.6, 1.4), "contrast")
    img = _enh(img, rng.uniform(0.5, 1.5), "saturation")
    img = hue_shift(img, rng.uniform(-0.15, 0.15))
    gains = (rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2))
    img = white_balance(img, gains)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.5)))
    if rng.random() < 0.5:
        img = add_noise(img, std=rng.uniform(0.0, 0.03), rng=rng)
    return img


CANDIDATES = {
    "none": aug_none,
    "brightness_contrast": aug_brightness_contrast,
    "saturation": aug_saturation,
    "hue": aug_hue,
    "white_balance": aug_white_balance,
    "blur": aug_blur,
    "noise": aug_noise,
    "combined": aug_combined,
}


# --------------------------------------------------------------------------- #
# Domain statistics & scoring
# --------------------------------------------------------------------------- #
def load_imgs(paths):
    return [Image.open(p).convert("RGB").resize((SIZE, SIZE)) for p in paths]


def rgb_means(imgs) -> np.ndarray:
    """Per-image mean RGB, shape (N, 3) in [0, 255]."""
    return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in imgs])


def coverage_score(aug_means: np.ndarray, target_means: np.ndarray) -> float:
    """Fraction of target images whose mean-RGB lies inside the *central* band
    (5th-95th percentile) of the augmented source distribution. Using
    percentiles instead of raw min/max ignores source outliers, so it actually
    discriminates between augmentations. Higher = the augmentation reshapes the
    source colour distribution to genuinely cover the target domain."""
    lo = np.percentile(aug_means, 5, axis=0)
    hi = np.percentile(aug_means, 95, axis=0)
    inside = np.all((target_means >= lo) & (target_means <= hi), axis=1)
    return float(inside.mean())


def center_distance(aug_means: np.ndarray, target_means: np.ndarray) -> float:
    """Euclidean distance between augmented-source and target mean colours."""
    return float(np.linalg.norm(aug_means.mean(axis=0) - target_means.mean(axis=0)))


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def save_aug_grid(name, fn, base_imgs, rng):
    cols = len(base_imgs)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.2, 4.6))
    for j, im in enumerate(base_imgs):
        axes[0, j].imshow(im)
        axes[0, j].axis("off")
        axes[1, j].imshow(fn(im, rng))
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=11)
    axes[1, 0].set_ylabel("augmented", fontsize=11)
    fig.suptitle(f"augmentation: {name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUTPUT_DIR, f"aug_{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_domain_compare(target_imgs, source_imgs, combined_imgs):
    rows = [
        ("TARGET (our camera)", target_imgs),
        ("SOURCE (clean_data)", source_imgs),
        ("SOURCE + combined aug", combined_imgs),
    ]
    cols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 2.2, len(rows) * 2.4))
    for i, (label, imgs) in enumerate(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(imgs):
                ax.imshow(imgs[j])
        axes[i, 0].set_title(label, loc="left", fontsize=11)
    fig.suptitle("Domain comparison: does the augmentation bridge the gap?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUTPUT_DIR, "domain_comparison.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    source_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.jpg")))
    target_paths = sorted(glob.glob(os.path.join(TARGET_DIR, "*.jpg")))
    if not source_paths or not target_paths:
        raise SystemExit("Could not find source/target images.")

    rng.shuffle(source_paths)
    sample_source_paths = source_paths[:200]  # subset for fast scoring
    source_imgs = load_imgs(sample_source_paths)
    target_imgs = load_imgs(target_paths)

    target_means = rgb_means(target_imgs)
    source_means = rgb_means(source_imgs)
    print("Target (our camera) mean RGB:", target_means.mean(axis=0).round(1))
    print("Source (clean_data) mean RGB:", source_means.mean(axis=0).round(1))
    print(f"Initial domain colour gap: {center_distance(source_means, target_means):.1f}\n")

    # Visualise each candidate on a few fixed images.
    vis_imgs = source_imgs[:6]
    results = []
    for name, fn in CANDIDATES.items():
        save_aug_grid(name, fn, vis_imgs, random.Random(SEED))
        # Score: apply augmentation to the sample set and measure target coverage.
        score_rng = random.Random(SEED + 1)
        aug_means = rgb_means([fn(im, score_rng) for im in source_imgs])
        cov = coverage_score(aug_means, target_means)
        spread = float(aug_means.std(axis=0).mean())  # colour diversity injected
        results.append((name, cov, spread))

    # Rank by how well the augmented source covers the target domain, breaking
    # ties toward more colour diversity (more robust to unseen casts).
    results.sort(key=lambda r: (-r[1], -r[2]))
    print(f"{'augmentation':<22}{'target_coverage':>16}{'colour_spread':>15}")
    print("-" * 53)
    for name, cov, spread in results:
        print(f"{name:<22}{cov:>15.0%}{spread:>15.1f}")

    best = results[0][0]
    print(f"\nRecommended augmentation: '{best}' "
          f"(covers {results[0][1]:.0%} of the target domain).")

    # Side-by-side domain comparison using the combined pipeline.
    combined_imgs = [aug_combined(im, random.Random(SEED + i)) for i, im in enumerate(source_imgs[:6])]
    cmp_path = save_domain_compare(target_imgs[:6], source_imgs[:6], combined_imgs)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as fh:
        fh.write("Augmentation search summary\n")
        fh.write(f"Target mean RGB: {target_means.mean(axis=0).round(1)}\n")
        fh.write(f"Source mean RGB: {source_means.mean(axis=0).round(1)}\n\n")
        fh.write(f"{'augmentation':<22}{'target_coverage':>16}{'colour_spread':>15}\n")
        for name, cov, spread in results:
            fh.write(f"{name:<22}{cov:>15.0%}{spread:>15.1f}\n")
        fh.write(f"\nRecommended: {best}\n")

    print(f"\nVisuals + summary saved to {os.path.relpath(OUTPUT_DIR, HERE)}/")
    print(f"  - per-augmentation grids: aug_<name>.png")
    print(f"  - domain comparison:      {os.path.basename(cmp_path)}")
    print(f"  - ranked table:           summary.txt")


if __name__ == "__main__":
    main()
