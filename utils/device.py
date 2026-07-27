"""
Device selection.

The training and inference code used to hardcode ``.cuda()``, which made the
project unrunnable on any machine without an NVIDIA GPU. Everything now routes
through :func:`get_device`, which prefers CUDA, falls back to Apple Silicon
(MPS), and finally to CPU.
"""

import torch


def get_device(preferred=None):
    """
    Resolve the compute device.

    Args:
        preferred: optional explicit device string ("cuda", "mps", "cpu").
            Falls back to autodetection when the request is unavailable.

    Returns:
        torch.device
    """
    if preferred:
        preferred = str(preferred).lower()
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cpu":
            return torch.device("cpu")
        print(f"Requested device '{preferred}' is unavailable; autodetecting instead.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device):
    """Human-readable device label for startup logging."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"
