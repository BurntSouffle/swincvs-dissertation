"""Memory probe: run 5 fwd+bwd steps at the config's batch size, print
peak VRAM. Used by launch_seed1.sh to fail fast on OOM before kicking
off a full training run.

Exit code 0 = OK; non-zero = OOM (or any other error). Stop the launch
script on non-zero exit.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

SWINCVS = Path("/workspace/SwinCVS")
os.chdir(SWINCVS)
sys.path.insert(0, str(SWINCVS))

from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from scripts.f_training_utils import build_optimizer, NativeScalerWithGradNormCount


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    cfg, exp = get_config(args.config)
    set_deterministic_behaviour(cfg.SEED)
    print(f"memory probe: config={args.config} bs={cfg.TRAIN.BATCH_SIZE} "
          f"steps={args.steps}")

    train_ds, _, _ = get_datasets(cfg)
    train_dl, _, _ = get_dataloaders(cfg, train_ds, train_ds, train_ds)

    model = build_model(cfg).cuda()
    optim = build_optimizer(cfg, model)
    scaler = NativeScalerWithGradNormCount()
    cw = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    crit = nn.BCEWithLogitsLoss(weight=cw).cuda()

    alpha = cfg.TRAIN.MULTICLASSIFIER_ALPHA
    beta = 1 - alpha

    torch.cuda.reset_peak_memory_stats()
    model.train()
    optim.zero_grad()

    for i, (x, y) in enumerate(train_dl):
        if i >= args.steps:
            break
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            a, b = model(x)
            loss = alpha * crit(a, y) + beta * crit(b, y)
        scaler(loss, optim, clip_grad=cfg.TRAIN.CLIP_GRAD,
               parameters=model.parameters(), create_graph=False,
               update_grad=True)
        optim.zero_grad()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  step {i+1}/{args.steps}: loss={loss.item():.4f} "
              f"peak_vram={peak_gb:.2f}GB")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = 100 * peak_gb / total_gb
    print()
    print(f"PROBE PASS — peak VRAM at bs={cfg.TRAIN.BATCH_SIZE}: "
          f"{peak_gb:.2f} GB / {total_gb:.1f} GB ({pct:.1f}%)")


if __name__ == "__main__":
    main()
