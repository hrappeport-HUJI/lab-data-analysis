"""Cellpose-SAM (cellpose >= 4) as a colony detector, zero-shot or fine-tuned.

Inputs a detector can be given (``CellposeDetector.input``):

* ``rgb``  - the frame as scanned
* ``diff`` - gray(frame - background), stretched; static plate texture and writing cancel out
* ``diff_rgb`` - 3 channels: diff, gray frame, gray previous frame (the time context the U-Net
  will also get, PLAN §5.3)
* ``scanlag`` - fine-tuned models (``CellposeDetector.finetuned``): ``train.dataset.model_channels``
  with the fixed normalization stored in the model's config.json

Cellpose-SAM is trained on objects about 30 px across. ScanLag colonies first appear at 3-5 px
across, so ``diameter`` rescales the tile by 30 / diameter before segmentation.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scanlag.detect.classical import rgb2gray_uint8
from scanlag.testset.gold import difference_image

INPUTS = ("rgb", "diff", "diff_rgb")


@dataclass
class CellposeDetector:
    input: str = "rgb"
    diameter: float | None = None  # None: no rescaling
    min_size: int = 15  # cellpose default; drops anything smaller (px, at original scale)
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    pretrained_model: str = "cpsam_v2"
    norm_limits: list | None = None  # fixed per-channel normalization (fine-tuned models)
    label: str | None = None
    _model: object = field(default=None, repr=False)
    _gpu: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)  # one eval at a time

    @property
    def name(self) -> str:
        if self.label:
            return f"{self.label}_min{self.min_size}"
        d = "auto" if self.diameter is None else f"{self.diameter:g}"
        return f"cellpose_{self.input}_d{d}_min{self.min_size}"

    def model(self):
        if self._model is None:
            from cellpose import models

            from scanlag.device import torch_device

            # Cellpose picks CUDA or the Apple GPU itself when gpu=True. Its default bfloat16 weights are
            # emulated, and very slow, on most CPUs (see scanlag.device.autocast).
            gpu = torch_device().type != "cpu"
            self._model = models.CellposeModel(gpu=gpu, pretrained_model=self.pretrained_model, use_bfloat16=gpu)
        return self._model

    @classmethod
    def finetuned(cls, model_dir: str | Path, **kwargs) -> CellposeDetector:
        """A model from ``scripts/train_cellpose.py``: its channels, normalization and scale."""
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        return cls(
            input="scanlag",
            diameter=cfg["diameter"],
            pretrained_model=str(Path(model_dir) / "model.pt"),
            norm_limits=cfg["norm_limits"],
            label=Path(model_dir).name,
            **{"min_size": 5, "cellprob_threshold": cfg.get("cellprob_threshold", 0.0), **kwargs},
        )

    def prepare(self, image: np.ndarray, background: np.ndarray, previous: np.ndarray) -> np.ndarray:
        if self.input == "scanlag":
            from scanlag.train.cellpose_finetune import normalize
            from scanlag.train.dataset import model_channels

            return normalize(model_channels(image, background, previous), self.norm_limits)
        if self.input == "rgb":
            return image.astype(np.float32) / 255
        diff = difference_image(image, background).astype(np.float32)
        if self.input == "diff":
            return diff
        if self.input == "diff_rgb":
            gray = [rgb2gray_uint8(x).astype(np.float32) / 255 for x in (image, previous)]
            return np.stack([diff, *gray], axis=-1)
        raise ValueError(f"input must be one of {INPUTS}")

    def segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        multichannel = images[0].ndim == 3
        with self._gpu:
            masks, _, _ = self.model().eval(
                images,
                channel_axis=-1 if multichannel else None,
                diameter=self.diameter,
                normalize=self.norm_limits is None,  # fine-tuned inputs are already normalized
                min_size=self.min_size,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
            )
        return [np.asarray(m, np.uint16) for m in masks]
