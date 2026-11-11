"""Fine-tune Cellpose-SAM on ScanLag tiles, with an ignore mask in the loss.

Cellpose's own ``train_seg`` has no ignore mask, so this is a small loop of its own that keeps
Cellpose's targets and loss: flows from ``masks_to_flows_gpu`` (x5) with MSE / 2, plus BCE on
the cell probability, both averaged over non-ignored pixels only.

Scale: Cellpose-SAM sees 8x8 px patches and was trained on objects about 30 px across, while
new colonies are 3-5 px across. Tiles are upscaled by ``scale`` (randomly jittered) before
training, and inference uses ``diameter = 30 / scale`` so Cellpose upscales the same way.

Normalization is fixed per channel (percentiles of the training set, stored with the model),
not per image, so an empty early tile isn't stretched until its noise looks like colonies.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fastremap
import numpy as np
import torch
import torch.nn.functional as F

from scanlag.device import autocast, torch_device
from scanlag.train.dataset import CHANNELS

BSIZE = 256  # fixed for Cellpose-SAM


@dataclass
class TrainConfig:
    base_model: str = "cpsam_v2"
    scale: float = 3.0
    scale_jitter: float = 0.2  # relative, uniform
    steps: int = 8000  # optimizer steps
    batch_size: int = 2  # per forward pass; batch 4 needs ~19 GB with ViT-L, more than the 4080 has
    accumulate: int = 2  # forward passes per optimizer step
    lr: float = 3e-5
    weight_decay: float = 0.1
    warmup: int = 300
    gold_weight: float = 5.0  # sampling weight of a gold tile relative to a MATLAB tile
    seed: int = 0


def norm_limits(x: np.ndarray, ignore: np.ndarray, n: int = 2_000_000, seed: int = 0) -> np.ndarray:
    """[channels, 2] (p1, p99.5) of each channel over non-ignored pixels."""
    rng = np.random.default_rng(seed)
    flat = x.reshape(-1, x.shape[-1])[~ignore.ravel()]
    sample = flat[rng.integers(0, len(flat), min(n, len(flat)))].astype(np.float32)
    lo, hi = np.percentile(sample, 1, axis=0), np.percentile(sample, 99.5, axis=0)
    return np.stack([lo, np.maximum(hi, lo + 1)], 1)


def normalize(x: np.ndarray, limits: np.ndarray) -> np.ndarray:
    limits = np.asarray(limits, np.float32)
    return (x.astype(np.float32) - limits[:, 0]) / (limits[:, 1] - limits[:, 0])


def augment(x, labels, ignore, scale: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random native crop, rot90/flip, upscale by ``scale`` to BSIZE x BSIZE."""
    c = min(math.ceil(BSIZE / scale), x.shape[0])
    y0, x0 = rng.integers(0, x.shape[0] - c + 1), rng.integers(0, x.shape[1] - c + 1)
    crop = [a[y0 : y0 + c, x0 : x0 + c] for a in (x, labels, ignore.astype(np.uint8))]
    k, flip = rng.integers(4), rng.random() < 0.5
    crop = [np.rot90(a, k) for a in crop]
    if flip:
        crop = [a[:, ::-1] for a in crop]
    size = (BSIZE, BSIZE)
    xi = cv2.resize(np.ascontiguousarray(crop[0]), size, interpolation=cv2.INTER_LINEAR)
    li = cv2.resize(np.ascontiguousarray(crop[1]).astype(np.int32), size, interpolation=cv2.INTER_NEAREST)
    ii = cv2.resize(np.ascontiguousarray(crop[2]), size, interpolation=cv2.INTER_NEAREST)
    return xi, li, ii.astype(bool)


def targets(labels: np.ndarray, device) -> np.ndarray:
    """[3, H, W]: cell probability, Y flow, X flow (Cellpose convention)."""
    from cellpose.dynamics import masks_to_flows_gpu

    labels = fastremap.renumber(labels.astype(np.int32), in_place=False)[0]
    out = np.zeros((3, *labels.shape), np.float32)
    if labels.max() > 0:
        out[0] = labels > 0
        out[1:] = masks_to_flows_gpu(labels, device=device)[0]
    return out


def masked_loss(y: torch.Tensor, lbl: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    n = valid.sum().clamp(min=1)
    flow = ((y[:, :2] - 5 * lbl[:, 1:]) ** 2).mean(1) / 2  # MSE over both components, /2 as Cellpose
    prob = F.binary_cross_entropy_with_logits(y[:, 2], lbl[:, 0], reduction="none")
    return ((flow + prob) * valid).sum() / n


def train(x, labels, ignore, source, out_dir: Path, cfg: TrainConfig, log=print) -> Path:
    from cellpose import models

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    limits = norm_limits(x, ignore)
    xn = normalize(x, limits)

    model = models.CellposeModel(gpu=torch_device().type != "cpu", pretrained_model=cfg.base_model, use_bfloat16=False)
    net, device = model.net, model.device
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / cfg.steps, 1))),
    )
    p = np.where(np.asarray(source) == "gold", cfg.gold_weight, 1.0)
    p /= p.sum()

    out_dir.mkdir(parents=True, exist_ok=True)
    history, t0, running = [], time.time(), []
    def batch():
        xs, ls, vs = [], [], []
        for i in rng.choice(len(xn), cfg.batch_size, p=p):
            s = cfg.scale * (1 + cfg.scale_jitter * (2 * rng.random() - 1))
            xi, li, ii = augment(xn[i], labels[i], ignore[i], s, rng)
            gain = 1 + 0.1 * (2 * rng.random(3) - 1)
            xi = xi * gain + 0.05 * (2 * rng.random(3) - 1)
            xs.append(xi.transpose(2, 0, 1))
            ls.append(targets(li, device))
            vs.append(~ii)
        return [torch.from_numpy(np.stack(a)).float().to(device) for a in (xs, ls, vs)]

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.accumulate):
            X, L, V = batch()
            with autocast(device):
                y = net(X)[0]
            loss = masked_loss(y.float(), L, V) / cfg.accumulate
            loss.backward()
            running.append(loss.item() * cfg.accumulate)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0 or step + 1 == cfg.steps:
            row = {"step": step + 1, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
            history.append(row)
            running = []
            log(f"step {row['step']}/{cfg.steps} loss={row['loss']:.4f} lr={row['lr']:.2e} {row['minutes']:.1f} min")
        if (step + 1) % 1000 == 0 or step + 1 == cfg.steps:
            net.save_model(str(out_dir / "model.pt"))
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    config = {
        **asdict(cfg),
        "channels": CHANNELS,
        "norm_limits": limits.tolist(),
        "diameter": 30.0 / cfg.scale,
        "n_tiles": int(len(x)),
        "n_gold_tiles": int((np.asarray(source) == "gold").sum()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir / "model.pt"
