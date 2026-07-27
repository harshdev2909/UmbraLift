"""
Generate synthetic low-light images for smoke-testing the pipeline without the
training dataset.

Produces crater-and-regolith-like greyscale scenes at very low exposure with
sensor noise, roughly matching the character of Permanently Shadowed Region
imagery: near-black, low contrast, noise-dominated.

These are NOT real lunar data. They exercise the training and inference paths;
they say nothing about enhancement quality. Quote metrics only from the real
dataset.

Usage:
    python -m utils.make_sample_images --count 40 --out data/train_data
"""

import argparse
import os

import numpy as np
from PIL import Image


def make_image(rng, size, exposure, noise):
    """Render one synthetic low-light scene as a HxWx3 uint8 array."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size

    # Broad illumination gradient, as if lit obliquely from one edge.
    scene = 0.35 + 0.25 * (1.0 - yy) + 0.10 * xx

    # A few craters: darker interiors with a bright rim on one side.
    for _ in range(rng.integers(3, 8)):
        cx, cy = rng.uniform(0.1, 0.9, size=2)
        radius = rng.uniform(0.05, 0.18)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inside = dist < radius
        rim = (dist >= radius * 0.85) & (dist < radius * 1.05)
        scene[inside] *= rng.uniform(0.45, 0.75)
        scene[rim] *= rng.uniform(1.15, 1.45)

    # Fine regolith texture.
    grain = rng.normal(0.0, 0.04, size=(size, size)).astype(np.float32)
    scene = scene + grain

    # Drop to a very low exposure, then add sensor noise. Noise dominating the
    # signal is what makes PSR enhancement hard.
    scene = np.clip(scene, 0.0, 1.0) * exposure
    scene = scene + rng.normal(0.0, noise, size=(size, size)).astype(np.float32)
    scene = np.clip(scene, 0.0, 1.0)

    rgb = np.repeat(scene[:, :, None], 3, axis=2)
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--size", type=int, default=256, help="Edge length (divisible by 8)")
    parser.add_argument("--out", default="data/train_data")
    parser.add_argument("--exposure", type=float, default=0.08, help="Peak brightness, 0-1")
    parser.add_argument("--noise", type=float, default=0.015, help="Sensor noise sigma")
    parser.add_argument("--seed", type=int, default=1143)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for i in range(args.count):
        img = make_image(rng, args.size, args.exposure, args.noise)
        Image.fromarray(img).save(os.path.join(args.out, f"psr_{i:03d}.png"))

    print(f"Wrote {args.count} images ({args.size}x{args.size}) to {args.out}")


if __name__ == "__main__":
    main()
