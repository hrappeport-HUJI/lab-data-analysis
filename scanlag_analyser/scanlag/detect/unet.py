"""U-Net colony detector (PLAN §5.3), a light alternative to Cellpose-SAM.

Network: U-Net with a ResNet-34 encoder (ImageNet init, segmentation_models_pytorch), about 24M
parameters. Two output channels:

* foreground logit: colony vs background
* distance map: distance to the colony edge, divided by the colony's own maximum, so every colony
  peaks at 1 in its centre whatever its size

Instances: the cores of colonies (distance > ``core``) seed a watershed of -distance inside the
foreground. Foreground pieces without a core become colonies of their own. Touching colonies
come out separated because each keeps its own core.

Input: the same 3 channels and fixed normalization as the fine-tuned Cellpose model
(``scanlag.train.dataset.model_channels``), upscaled by ``scale`` so 3-5 px colonies span more
than one pixel at the network's finest decoder level.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from scanlag.device import autocast, torch_device

PAD = 32  # the encoder downsamples by 32


def build_unet(encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet34", encoder_weights=encoder_weights, in_channels=3, classes=2)


def distance_target(labels: np.ndarray) -> np.ndarray:
    """Per-colony distance to the colony edge, scaled to [0, 1] within each colony."""
    out = np.zeros(labels.shape, np.float32)
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        obj = labels[sl] == i
        d = ndi.distance_transform_edt(obj)
        out[sl][obj] = d[obj] / d.max()
    return out


def instances(fg_prob: np.ndarray, dist: np.ndarray, fg: float = 0.4, core: float = 0.7, min_size: int = 5,
              smooth: float = 1.0) -> np.ndarray:
    """Higher ``core`` splits less. The predicted distance peaks at about 0.75-0.85, not 1, so
    ``core`` >= 0.85 leaves almost every connected foreground piece whole."""
    mask = fg_prob > fg
    if smooth:
        dist = ndi.gaussian_filter(dist, smooth)
    markers, n = ndi.label(mask & (dist > core))
    labels = watershed(-dist, markers, mask=mask)
    rest, m = ndi.label(mask & (labels == 0))  # foreground without a core
    labels = np.where(rest > 0, rest + n, labels)
    sizes = np.bincount(labels.ravel())
    small = sizes < min_size
    small[0] = False
    labels[small[labels]] = 0
    return labels.astype(np.uint16)


@dataclass
class UNetDetector:
    model_dir: Path
    fg: float = 0.4
    core: float = 0.7
    min_size: int = 5
    _net: object = field(default=None, repr=False)
    _gpu: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)  # one forward pass at a time

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())

    @property
    def name(self) -> str:
        return f"{self.model_dir.name}_core{self.core:g}"

    def net(self) -> torch.nn.Module:
        if self._net is None:
            net = build_unet(encoder_weights=None)
            net.load_state_dict(torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=True))
            self._net = net.to(torch_device()).eval()
        return self._net

    def prepare(self, image: np.ndarray, background: np.ndarray, previous: np.ndarray) -> np.ndarray:
        from scanlag.train.cellpose_finetune import normalize
        from scanlag.train.dataset import model_channels

        return normalize(model_channels(image, background, previous), self.config["norm_limits"])

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Foreground probability and distance map at the input's (native) resolution.

        Thread-safe. The lock covers only queueing the work on the GPU: the input is laid out in
        page-locked memory before it, and the result is awaited after it, so other threads can
        prepare their input and queue their own pass meanwhile.
        """
        cuda = torch_device().type == "cuda"  # page-locked memory and events are CUDA only
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), np.float32)[None])
        if cuda:
            t = t.pin_memory()
        with self._gpu:
            net = self.net()
            device = next(net.parameters()).device
            y = self._forward(net, t.to(device, non_blocking=True), x.shape[:2])
            out = torch.empty(y.shape, pin_memory=cuda)
            out.copy_(y, non_blocking=cuda)
            if cuda:
                done = torch.cuda.Event()
                done.record()
        if cuda:
            done.synchronize()
        out = out.numpy()
        return out[0], out[1]

    def _forward(self, net, t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        s = self.config["scale"]
        H, W = round(h * s), round(w * s)
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        t = F.pad(t, (0, -W % PAD, 0, -H % PAD), mode="reflect")
        with autocast(t.device):
            y = net(t)[..., :H, :W].float()
        if y.device.type == "mps" and (H % h or W % w):
            y = y.cpu()  # the Apple GPU's area resampling only takes whole-number factors
        y = F.interpolate(y, size=(h, w), mode="area")[0]
        return torch.stack([torch.sigmoid(y[0]), torch.sigmoid(y[1])])

    def segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [instances(*self.predict(x), self.fg, self.core, self.min_size) for x in images]


def load_detector(model_dir: str | Path, **kwargs):
    """The right detector for a model directory written by scripts/train_*.py."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if cfg.get("type") == "unet":
        return UNetDetector(model_dir, **kwargs)
    from scanlag.detect.cellpose_backend import CellposeDetector

    return CellposeDetector.finetuned(model_dir, **kwargs)
