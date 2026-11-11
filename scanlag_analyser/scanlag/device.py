"""Where the networks run: an NVIDIA GPU (CUDA), else an Apple-silicon GPU (MPS), else the CPU.

The environment variable SCANLAG_DEVICE (cuda, mps or cpu) overrides the choice, e.g. to compare
results between devices or to fall back to the CPU if a step fails on the Apple GPU.
"""

from __future__ import annotations

import os

import torch

DEVICES = ("cuda", "mps", "cpu")


def torch_device() -> torch.device:
    if name := os.environ.get("SCANLAG_DEVICE", "").strip().lower():
        if name not in DEVICES:
            raise ValueError(f"SCANLAG_DEVICE={name!r}: use one of {', '.join(DEVICES)}")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device) -> torch.autocast:
    """Mixed precision for the networks' forward pass: bfloat16 on CUDA (as the models were trained),
    float16 on the Apple GPU (bfloat16 needs macOS 14 there and is not faster), none on the CPU, where
    bfloat16 is emulated on most processors and was measured ~1000x slower than float32."""
    dtype = torch.float16 if device.type == "mps" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type != "cpu")
