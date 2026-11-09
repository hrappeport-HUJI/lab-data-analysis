"""Train the U-Net colony detector (``scanlag.detect.unet``) on the same tiles as Cellpose.

Losses, averaged over non-ignored pixels: BCE + soft Dice on the foreground, and 5 x MSE on the
per-colony distance map. Augmentation and fixed normalization are shared with the Cellpose
fine-tuning (``cellpose_finetune.augment`` / ``norm_limits``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scanlag.detect.unet import build_unet, distance_target
from scanlag.device import autocast, torch_device
from scanlag.train.cellpose_finetune import augment, norm_limits, normalize
from scanlag.train.dataset import CHANNELS


@dataclass
class UNetConfig:
    scale: float = 2.0
    scale_jitter: float = 0.2
    steps: int = 6000
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    gold_weight: float = 5.0
    dist_weight: float = 5.0
    seed: int = 0


def unet_loss(y: torch.Tensor, fg: torch.Tensor, dist: torch.Tensor, valid: torch.Tensor, dist_weight: float) -> torch.Tensor:
    n = valid.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(y[:, 0], fg, reduction="none") * valid).sum() / n
    p = torch.sigmoid(y[:, 0]) * valid
    dice = 1 - (2 * (p * fg).sum() + 1) / (p.sum() + (fg * valid).sum() + 1)
    mse = (((torch.sigmoid(y[:, 1]) - dist) ** 2) * valid).sum() / n
    return bce + dice + dist_weight * mse


def train(x, labels, ignore, source, out_dir: Path, cfg: UNetConfig, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    limits = norm_limits(x, ignore)
    xn = normalize(x, limits)
    device = torch_device()
    net = build_unet().to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / cfg.steps, 1)))
    )
    p = np.where(np.asarray(source) == "gold", cfg.gold_weight, 1.0)
    p /= p.sum()

    out_dir.mkdir(parents=True, exist_ok=True)
    history, running, t0 = [], [], time.time()
    for step in range(cfg.steps):
        xs, fgs, ds, vs = [], [], [], []
        for i in rng.choice(len(xn), cfg.batch_size, p=p):
            s = cfg.scale * (1 + cfg.scale_jitter * (2 * rng.random() - 1))
            xi, li, ii = augment(xn[i], labels[i], ignore[i], s, rng)
            xi = xi * (1 + 0.1 * (2 * rng.random(3) - 1)) + 0.05 * (2 * rng.random(3) - 1)
            xs.append(xi.transpose(2, 0, 1))
            fgs.append(li > 0)
            ds.append(distance_target(li))
            vs.append(~ii)
        X, FG, D, V = (torch.from_numpy(np.stack(a)).float().to(device) for a in (xs, fgs, ds, vs))
        with autocast(device):
            y = net(X)
        loss = unet_loss(y.float(), FG, D, V, cfg.dist_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            row = {"step": step + 1, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
            history.append(row)
            running = []
            log(f"step {row['step']}/{cfg.steps} loss={row['loss']:.4f} lr={row['lr']:.2e} {row['minutes']:.1f} min")
        if (step + 1) % 1000 == 0 or step + 1 == cfg.steps:
            torch.save(net.state_dict(), out_dir / "model.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    config = {"type": "unet", "encoder": "resnet34", **asdict(cfg), "channels": CHANNELS,
              "norm_limits": limits.tolist(), "n_tiles": int(len(x)),
              "n_gold_tiles": int((np.asarray(source) == "gold").sum())}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir / "model.pt"
