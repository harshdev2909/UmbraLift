"""
Testing script for UmbraLift.

Enhances every image under a directory tree and writes the results to a
mirrored `result/` tree.
"""

import os
import argparse
import glob
import time

import numpy as np
import torch
import torchvision
from PIL import Image

import model
from utils.device import get_device, describe

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_network(snapshot_path, device):
    """
    Build the network and load weights once.

    Previously this ran per image, re-reading the checkpoint from disk for
    every file processed.
    """
    net = model.umbra_lift_net().to(device)
    net.load_state_dict(torch.load(snapshot_path, map_location=device, weights_only=True))
    net.eval()
    return net


def enhance(image_path, net, device, output_root=None, input_root=None):
    """
    Enhance a single low-light image and save it to the result tree.

    Args:
        image_path (str): Path to the input low-light image.
        net: The loaded network.
        device: Compute device.
        output_root (str, optional): Root for outputs. Defaults to replacing
            'test_data' with 'result' in the input path.
        input_root (str, optional): Root the relative output path is built from.

    Returns:
        str: Path the enhanced image was written to.
    """
    data_lowlight = Image.open(image_path).convert("RGB")

    # The network pools three times with stride 2 and unpools using the stored
    # indices, so both dimensions must be divisible by 8.
    width = (data_lowlight.size[0] // 8) * 8
    height = (data_lowlight.size[1] // 8) * 8
    if width == 0 or height == 0:
        raise ValueError(f"Image too small to process: {image_path} ({data_lowlight.size})")
    if (width, height) != data_lowlight.size:
        data_lowlight = data_lowlight.resize((width, height))

    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).to(device).unsqueeze(0)

    enhanced_image, _ = net(data_lowlight)

    if output_root and input_root:
        rel = os.path.relpath(image_path, input_root)
        result_path = os.path.join(output_root, rel)
    else:
        result_path = image_path.replace("test_data", "result")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path


def main():
    parser = argparse.ArgumentParser(description='UmbraLift Testing Script')
    parser.add_argument('--lowlight_images_path', type=str, default="data/test_data/",
                        help='Directory of low-light test images (searched recursively)')
    parser.add_argument('--pretrain_snapshot', type=str, default="snapshots/model-best.pth",
                        help='Pretrained model snapshot')
    parser.add_argument('--output_path', type=str, default=None,
                        help="Output directory (default: input path with 'test_data' replaced by 'result')")
    parser.add_argument('--device', type=str, default=None, help='Force a device: cuda, mps or cpu')
    config = parser.parse_args()

    if not os.path.isdir(config.lowlight_images_path):
        raise SystemExit(f"Input directory not found: {config.lowlight_images_path}")
    if not os.path.isfile(config.pretrain_snapshot):
        raise SystemExit(f"Checkpoint not found: {config.pretrain_snapshot}")

    device = get_device(config.device)
    print(f"Running on {describe(device)}")

    net = load_network(config.pretrain_snapshot, device)

    # Recursive walk, so both flat directories and the documented
    # test_data/<set>/ layout work.
    image_paths = sorted(
        p for p in glob.glob(os.path.join(config.lowlight_images_path, "**", "*"), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise SystemExit(f"No images found under {config.lowlight_images_path}")

    print(f"Found {len(image_paths)} image(s)")
    start = time.time()
    with torch.no_grad():
        for i, image_path in enumerate(image_paths, 1):
            result_path = enhance(image_path, net, device,
                                  output_root=config.output_path,
                                  input_root=config.lowlight_images_path)
            print(f"  [{i}/{len(image_paths)}] {os.path.basename(image_path)} -> {result_path}")
    elapsed = time.time() - start

    print(f"Enhanced {len(image_paths)} image(s) in {elapsed:.2f}s "
          f"({elapsed / len(image_paths) * 1000:.1f} ms/image)")


if __name__ == '__main__':
    main()
