"""
This module provides functionality for loading and processing low-light images for training.
It includes utilities for dataset creation and image preprocessing.
"""

import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_SEED = 1143


def populate_train_list(lowlight_images_path, seed=DEFAULT_SEED):
    """
    Collect image file paths under a directory tree, shuffled.

    Only files with a recognised image extension are returned, so stray
    entries such as .DS_Store or notes.txt do not reach the loader and blow up
    at read time.

    Args:
        lowlight_images_path (str): Directory containing low-light images.
        seed (int): Seed for the shuffle, for reproducible ordering.

    Returns:
        list: Shuffled list of image file paths.
    """
    file_paths_and_names = []

    for dirpath, _dirnames, filenames in os.walk(lowlight_images_path):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            file_paths_and_names.append(os.path.join(dirpath, filename))

    # Local RNG rather than seeding the global one at import time.
    random.Random(seed).shuffle(file_paths_and_names)

    return file_paths_and_names


class lowlight_loader(data.Dataset):
    """
    Dataset of low-light images, resized to a fixed square size.

    Args:
        lowlight_images_path (str): Directory containing low-light images.
        size (int): Edge length images are resized to. Must be divisible by 8,
            since the network pools three times and unpools with stored indices.
        max_images (int, optional): Cap on the number of images used. ``None``
            (the default) uses the whole dataset.
        seed (int): Seed for shuffling and subsampling.
    """

    def __init__(self, lowlight_images_path, size=128, max_images=None, seed=DEFAULT_SEED):
        if size % 8 != 0:
            raise ValueError(
                f"size must be divisible by 8 (got {size}); the network pools "
                "three times with stride 2 and unpools with stored indices."
            )

        self.train_list = populate_train_list(lowlight_images_path, seed=seed)
        self.size = size

        if not self.train_list:
            raise RuntimeError(
                f"No images found under '{lowlight_images_path}'. Expected files with "
                f"extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )

        total = len(self.train_list)

        # Previously this silently dropped the dataset to 200 images whenever
        # more than 1000 were present, which quietly discarded most of the
        # training data. Subsampling is now opt-in and always reported.
        if max_images is not None and total > max_images:
            self.train_list = random.Random(seed).sample(self.train_list, max_images)
            print(f"Total training examples: {total} (subsampled to {max_images})")
        else:
            print(f"Total training examples: {total}")

        self.data_list = self.train_list

    def __getitem__(self, index):
        """
        Load and preprocess a single image.

        Returns:
            torch.Tensor: Image in CHW format, normalised to [0, 1].
        """
        data_lowlight_path = self.data_list[index]

        # Force RGB so greyscale or RGBA source images still yield 3 channels.
        data_lowlight = Image.open(data_lowlight_path).convert("RGB")
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.Resampling.LANCZOS)
        data_lowlight = np.asarray(data_lowlight) / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
